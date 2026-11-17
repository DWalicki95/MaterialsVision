"""Registry of per-series filename parsers.

The AS, VAB/K and PIR series use unrelated filename conventions: AS
encodes magnification in the name and never has a cross-section; VAB/K
never encodes magnification but always repeats formulation+cross-
section twice; PIR encodes neither formulation nor cross-section, only
a folder code that ``PIRProfile.FOLDERS`` resolves. A single regex
covering them all would be unreadable and fragile, so each series gets
its own ``SeriesProfile`` implementation bundling:
the filename parser, the sidecar path convention, and a lightweight
series-consistency check.

Note on ``parse()`` signature: it takes the run-scoped
``IssueCollector`` as a required keyword argument, even though a bare
``parse(stem)`` would look tidier. ``ParsedName.cross_section`` is
guaranteed canonical, and canonicalizing it can involve a fuzzy match
that must be reported rather than applied silently (see
``cross_section.py``). ``ASProfile`` has no cross-section to resolve
and ignores the collector, but must still accept it to stay
interchangeable through ``SeriesProfile``.
"""
import re
from abc import ABC, abstractmethod
from pathlib import Path

from data_prep.inventory.cross_section import (canonicalize_cross_section,
                                               lookup_cross_section)
from data_prep.inventory.issues import (FilenameParseError, IssueCollector,
                                        IssueLevel)
from data_prep.inventory.models import ParsedName

# Label Studio prepends an 8-hex-character task hash; every filename seen
# in the current export has it, but it is treated as optional noise
# rather than assumed, matching the convention already used in
# data_prep/split_dataset_into_subsets.py.
_LS_PREFIX = r"(?:[0-9a-f]{8}-)?"


class SeriesProfile(ABC):
    """One filename convention + its sidecar mapping rule.

    Attributes
    ----------
    name : str
        Series identifier, e.g. ``"AS"`` or ``"VAB"``.
    """

    name: str

    @abstractmethod
    def parse(
        self,
        filename: str,
        *,
        collector: IssueCollector,
        fuzzy_cutoff: float = 0.85,
    ) -> ParsedName:
        """Parse a filename (without extension) into a ``ParsedName``.

        Parameters
        ----------
        filename : str
            Filename stem, as it appears in the Label Studio export's
            ``data.image`` field (basename, no extension).
        collector : IssueCollector
            Collector for INFO-level issues raised while resolving
            ambiguous tokens (fuzzy cross-section matches).
        fuzzy_cutoff : float, optional
            Passed through to cross-section canonicalization.

        Returns
        -------
        ParsedName

        Raises
        ------
        FilenameParseError
            If ``filename`` does not match this series' convention.
        """

    @abstractmethod
    def sidecar_candidates(self, parsed: ParsedName) -> tuple[Path, ...]:
        """Return relative sidecar paths to try, in order.

        Each returned path is relative and is joined against every
        configured ``sem_metadata_dirs`` root in turn by
        ``sem_sidecar.find_sidecar``.

        Parameters
        ----------
        parsed : ParsedName

        Returns
        -------
        tuple of Path
        """

    @abstractmethod
    def detect_series(self, filename: str) -> bool:
        """Check whether ``filename`` looks like this series' naming.

        This is only a consistency check against the series declared in
        the configuration - it never decides which profile to use.

        Parameters
        ----------
        filename : str

        Returns
        -------
        bool
        """


class ASProfile(SeriesProfile):
    """Filename convention for the AS series (TM3000, Roboflow-exported).

    Pattern::

        [<ls_hash>-]<FORMULATION>_<MAGNIFICATION>[_<SAMPLE>]_jpg.rf.<hash>_image

    ``SAMPLE`` is kept as a string because six real filenames use
    non-numeric or multi-segment tokens (``AS18_40_-``, ``AS19_50``
    lacking a sample entirely, ``AS25_40_1-5``, ``AS1A_40_21a``,
    ``AS4_40_500_1``, ``AS4_40_500_2``); none of them prevent parsing
    the formulation or magnification, so none are treated as fatal.
    """

    name = "AS"

    _OUTER_RE = re.compile(
        rf"^{_LS_PREFIX}(?P<core>.+)_jpg\.rf\.(?P<rf_hash>[0-9a-f]+)_image$"
    )
    _CORE_RE = re.compile(
        r"^(?P<formulation>AS\d+[A-Z]?)_(?P<mag>\d+)"
        r"(?:_(?P<sample>.+))?$"
    )
    _HINT_RE = re.compile(rf"^{_LS_PREFIX}AS\d")

    def parse(
        self,
        filename: str,
        *,
        collector: IssueCollector,
        fuzzy_cutoff: float = 0.85,
    ) -> ParsedName:
        outer = self._OUTER_RE.match(filename)
        if not outer:
            raise FilenameParseError(
                f"AS filename does not match "
                f"'<formulation>_<mag>[_<sample>]_jpg.rf.<hash>_image': "
                f"{filename}"
            )
        core = outer.group("core")
        inner = self._CORE_RE.match(core)
        if not inner:
            raise FilenameParseError(
                f"AS core does not match "
                f"'<formulation>_<mag>[_<sample>]': {core}"
            )

        formulation = inner.group("formulation")
        magnification = int(inner.group("mag"))
        sample = inner.group("sample")
        is_nonstandard = sample is None or not re.fullmatch(
            r"\d+", sample
        )

        return ParsedName(
            image_id=core,
            series=self.name,
            material="AS",
            formulation=formulation,
            cross_section=None,
            magnification_from_name=magnification,
            sample_id=sample,
            source_filename=filename,
            is_nonstandard=is_nonstandard,
            cross_section_redundancy_ok=None,
        )

    def sidecar_candidates(self, parsed: ParsedName) -> tuple[Path, ...]:
        return (Path(parsed.formulation) / f"{parsed.image_id}.txt",)

    def detect_series(self, filename: str) -> bool:
        return bool(self._HINT_RE.match(filename))


class VABProfile(SeriesProfile):
    """Filename convention for the VAB/K series (SU8000).

    Pattern::

        [<ls_hash>-]<F1>_<C1>_<F2>_<C2>_m<NUM>

    where ``F1``/``F2`` are ``VAB<n>`` or ``K<n>`` and ``C1``/``C2`` are
    cross-section tokens (the second occurrence is often abbreviated,
    e.g. ``K3_rownolegle_K3_r_m004``). Magnification is never encoded
    here - it comes exclusively from the SEM sidecar.
    """

    name = "VAB"

    _VAB_RE = re.compile(
        rf"^{_LS_PREFIX}(?P<f1>(?:VAB|K)\d+)_(?P<c1>[A-Za-z]+)_"
        rf"(?P<f2>(?:VAB|K)\d+)_(?P<c2>[A-Za-z]+)_m(?P<num>\d+)$"
    )
    _HINT_RE = re.compile(rf"^{_LS_PREFIX}(?:VAB|K)\d")

    def parse(
        self,
        filename: str,
        *,
        collector: IssueCollector,
        fuzzy_cutoff: float = 0.85,
    ) -> ParsedName:
        match = self._VAB_RE.match(filename)
        if not match:
            raise FilenameParseError(
                f"VAB/K filename does not match "
                f"'<F1>_<C1>_<F2>_<C2>_m<NUM>': {filename}"
            )

        f1 = match.group("f1")
        c1_raw = match.group("c1")
        f2 = match.group("f2")
        c2_raw = match.group("c2")
        num = match.group("num")

        formulation = f1
        material = "K" if f1.upper().startswith("K") else "VAB"
        cross_section = canonicalize_cross_section(
            c1_raw,
            fuzzy_cutoff=fuzzy_cutoff,
            collector=collector,
            image_ref=filename,
        )
        image_id = f"{formulation}_{cross_section}_m{num}"

        redundancy_ok = self._check_redundancy(
            f1, cross_section, f2, c2_raw
        )
        if not redundancy_ok:
            collector.add(
                IssueLevel.WARNING,
                "vab_redundancy_mismatch",
                image_id,
                f"redundant name segment mismatch: "
                f"'{f1}_{c1_raw}' vs '{f2}_{c2_raw}'",
            )

        return ParsedName(
            image_id=image_id,
            series=self.name,
            material=material,
            formulation=formulation,
            cross_section=cross_section,
            magnification_from_name=None,
            sample_id=num,
            source_filename=filename,
            is_nonstandard=False,
            cross_section_redundancy_ok=redundancy_ok,
        )

    @staticmethod
    def _check_redundancy(
        f1: str, canonical_c1: str, f2: str, c2_raw: str
    ) -> bool:
        """Check the redundant second name segment against the first.

        Parameters
        ----------
        f1 : str
            First formulation token, as written in the filename.
        canonical_c1 : str
            Already-canonicalized cross-section of the first segment.
        f2 : str
            Second formulation token.
        c2_raw : str
            Second (often abbreviated) cross-section token.

        Returns
        -------
        bool
            True if the formulation matches case-insensitively and the
            second cross-section token resolves (exactly, no fuzzy) to
            the same canonical value as the first.
        """
        if f1.lower() != f2.lower():
            return False
        return lookup_cross_section(c2_raw) == canonical_c1

    def sidecar_candidates(self, parsed: ParsedName) -> tuple[Path, ...]:
        match = self._VAB_RE.match(parsed.source_filename)
        if not match:
            # parse() already succeeded for this source_filename, so
            # this can only happen if callers pass a mismatched pair.
            raise FilenameParseError(
                f"source_filename no longer matches VAB pattern: "
                f"{parsed.source_filename}"
            )
        f1 = match.group("f1")
        c1_raw = match.group("c1")
        f2 = match.group("f2")
        c2_raw = match.group("c2")
        num = match.group("num")
        return (
            Path(f"{f1} {c1_raw}") / f"{f2} {c2_raw}_m{num}.txt",
        )

    def detect_series(self, filename: str) -> bool:
        return bool(self._HINT_RE.match(filename))


class PIRProfile(SeriesProfile):
    """Filename convention for the PIR series (rigid PIR foams with a
    cocoa-husk filler, TM3000, 2025).

    Pattern::

        [<ls_hash>-]<CODE>_<MAG>[<sep><N>|<suffix>]

    ``CODE`` is a two-digit folder code (``01`` ... ``08``) and is the
    only place the formulation and cross-section are recorded, so they
    come from ``FOLDERS`` rather than from the name. ``sep`` is ``-``
    or ``_`` depending on the folder. A base image has no ``N``; one
    folder spells its base image ``08_40szw`` instead, which is kept
    as a nonstandard name for the same base image.

    ``MAG`` is not reliable: ``03_40_1`` and ``03_40_2`` were taken at
    50x, as their sidecars and burned-in data panels both show. The
    sidecar wins whenever there is one; the images without a sidecar
    were all checked to be at the 40x their name states.
    """

    name = "PIR"

    # Folder code -> (sidecar folder, formulation, canonical cross-
    # section). Formulations name the filler content in php after the
    # sidecar's own Directory field (SZW, SZW-5, SZW-15, SZW-25), which
    # is authoritative where the folder name disagrees with it: folder
    # 08 says "z_napelniaczem" but its sidecars point at SZW (no
    # filler), the perpendicular pair of folder 01. Folder 05 is a
    # byte-for-byte copy of 02 and is excluded in the configuration, so
    # it has no entry here.
    FOLDERS: dict[str, tuple[str, str, str]] = {
        "01": ("sztywne_pianka_bez_napelniacza_rownolegle", "PIR0",
               "rownolegly"),
        "02": ("sztywne_prostopadle_15php_kakao", "PIR15", "prostopadly"),
        "03": ("sztywne_prostopadle_25php_kakao", "PIR25", "prostopadly"),
        "04": ("sztywne_prostopadle_5proc_napelniacza_kakao", "PIR5",
               "prostopadly"),
        "06": ("sztywne_rownolegle_25php_kakao", "PIR25", "rownolegly"),
        "07": ("sztywne_rownolegle_5php_kakao", "PIR5", "rownolegly"),
        "08": ("sztywne_z_napelniaczem_luski_kakao", "PIR0",
               "prostopadly"),
    }

    _PIR_RE = re.compile(
        rf"^{_LS_PREFIX}(?P<code>\d{{2}})_(?P<mag>\d+)"
        r"(?:(?P<sep>[-_])(?P<num>\d+)|(?P<suffix>[a-z]+))?$"
    )
    _HINT_RE = re.compile(rf"^{_LS_PREFIX}\d{{2}}_\d")

    def parse(
        self,
        filename: str,
        *,
        collector: IssueCollector,
        fuzzy_cutoff: float = 0.85,
    ) -> ParsedName:
        match = self._match(filename)
        code = match.group("code")
        if code not in self.FOLDERS:
            raise FilenameParseError(
                f"PIR folder code {code!r} has no FOLDERS entry "
                f"(known: {sorted(self.FOLDERS)}): {filename}"
            )
        _, formulation, cross_section = self.FOLDERS[code]
        num = match.group("num")
        sample = num if num is not None else "0"

        return ParsedName(
            image_id=f"{formulation}_{cross_section}_{sample}",
            series=self.name,
            material="PIR",
            formulation=formulation,
            cross_section=cross_section,
            magnification_from_name=int(match.group("mag")),
            sample_id=sample,
            source_filename=filename,
            is_nonstandard=match.group("suffix") is not None,
            cross_section_redundancy_ok=None,
        )

    def sidecar_candidates(self, parsed: ParsedName) -> tuple[Path, ...]:
        match = self._match(parsed.source_filename)
        folder, _, _ = self.FOLDERS[match.group("code")]
        stem = match.group("mag")
        if match.group("num") is not None:
            stem += match.group("sep") + match.group("num")
        return (Path(folder) / f"{stem}.txt",)

    def detect_series(self, filename: str) -> bool:
        return bool(self._HINT_RE.match(filename))

    def _match(self, filename: str) -> re.Match:
        """Match ``filename`` against the PIR pattern or raise.

        Parameters
        ----------
        filename : str

        Returns
        -------
        re.Match

        Raises
        ------
        FilenameParseError
            If ``filename`` does not match the PIR convention.
        """
        match = self._PIR_RE.match(filename)
        if not match:
            raise FilenameParseError(
                f"PIR filename does not match "
                f"'<code>_<mag>[<sep><n>|<suffix>]': {filename}"
            )
        return match


PROFILES: dict[str, SeriesProfile] = {
    "AS": ASProfile(),
    "VAB": VABProfile(),
    "PIR": PIRProfile(),
}


def get_profile(series: str) -> SeriesProfile:
    """Look up the ``SeriesProfile`` for a configured series name.

    Parameters
    ----------
    series : str
        Series identifier as declared in the source configuration
        (e.g. ``"AS"``, ``"VAB"``).

    Returns
    -------
    SeriesProfile

    Raises
    ------
    KeyError
        If no profile is registered for ``series``.
    """
    try:
        return PROFILES[series]
    except KeyError:
        raise KeyError(
            f"No SeriesProfile registered for series '{series}'. "
            f"Known series: {sorted(PROFILES)}"
        ) from None
