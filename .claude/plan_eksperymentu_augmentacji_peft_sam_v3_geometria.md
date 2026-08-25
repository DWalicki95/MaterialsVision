# Plan eksperymentu augmentacji danych dla dostrajania Micro-SAM / PEFT-SAM

**Zadanie badawcze:** automatyczna segmentacja instancyjna porów na obrazach SEM wiskoelastycznych pianek poliuretanowych.
**Model:** Micro-SAM (Segment Anything Model zaadaptowany do mikroskopii) dostrajany metodą LoRA w ramach frameworku PEFT-SAM, z dekoderem instancyjnym AIS.
**Jednostka wejściowa:** cały obraz SEM w rozdzielczości źródłowej 1280 × 960 (AS) albo 1280 × 890 po docięciu paska informacyjnego (K, VAB); nie stosuje się treningu na wycinkach (patchach). Szczegóły geometrii w części III.1.
**Cel nadrzędny (podwójny):** (1) uzyskanie najlepszego możliwego modelu do praktycznego użycia oraz (2) naukowa, publikowalna atrybucja wkładu poszczególnych rodzin augmentacji. Oba cele są równorzędne. Dlatego plan zawiera prawdziwy baseline bez augmentacji (B0) oraz pełną ablację leave-one-family-out z FULL.
**Status:** dokument wykonawczy, kompletny i samodzielny. **Na dzień 2026-08-10 nie zawiera otwartych kwestii do rozstrzygnięcia** — wszystkie decyzje metodologiczne są podjęte i zamrożone (część XVI.1), a pozostałe wielkości mają zamrożoną procedurę wyznaczenia (część XVI.2).

**Uwaga o danych.** Inwentaryzacja została wykonana (2026-08-10) i jej wyniki są wpisane do dokumentu. Zbiór liczy **707 obrazów w 31 formulacjach**, zgrupowanych w **3 rodziny pianek** (AS, K, VAB), rozpoznawalne z prefiksu nazwy pliku. W całym dokumencie „formulacja" oznacza pojedynczy wariant materiałowy (jedna synteza), a „rodzina" — materiał nadrzędny grupujący formulacje.

**Nazewnictwo — jedno źródło prawdy.** Rodzinie pianki odpowiada w manifeście kolumna **`material`** (wartości `AS`, `K`, `VAB`). W tekście planu używamy słowa „rodzina", w kodzie i we wszystkich wyrażeniach technicznych — `material`. **Nie tworzy się osobnej kolumny `family`**; wcześniejsze wystąpienia tej nazwy zostały w całym dokumencie zastąpione przez `material` (uzasadnienie: kolumna `material` istniała w manifeście przed powstaniem tego zapisu i duplikowanie jej pod drugą nazwą dałoby dwa źródła prawdy). Szczegóły w części III.1.

**Aktualizacja 2026-08-10 (A) — co zmieniła inwentaryzacja skali.** Zdjęcia powstały na **dwóch mikroskopach** o różnej kalibracji, więc nominalne powiększenie przestało być wiarygodną miarą skali (to samo „40×" oznacza 3.24 µm/px na jednym mikroskopie i 2.48 µm/px na drugim). W konsekwencji: (1) zmienną skali jest `pixel_size_um`, a nie powiększenie; (2) stratyfikacja i wszystkie przekroje raportowania idą po `material × scale_bin`, nie po powiększeniu; (3) `q_max` dla F2 jest skalibrowane na 1.30 (część V.2); (4) sześć obrazów-zbliżeń jest oznaczonych jako `scale_outlier` i wyłączonych z oceny; (5) bramka priorytetu E2 została zaliczona.

**Aktualizacja 2026-08-10 (B) — co zmieniła weryfikacja geometrii.** Pierwotne założenie o rozdzielczości źródłowej **960 × 1080 było błędne**, a wyprowadzony z niego współczynnik preprocessingu ×0.948 — również. Rzeczywiste pliki mają **1280 × 960**, przy czym w serii AS pasek informacyjny został już usunięty przed anotacją, a w seriach K i VAB **wciąż jest obecny** i zajmuje dolne 70 wierszy. W konsekwencji: (1) docięcie paska dla K i VAB staje się zamrożonym krokiem preprocessingu (część IV.1), bo pasek jest idealnie skorelowany z mikroskopem M2 i bez docięcia byłby artefaktem nierozróżnialnym od efektu materiałowego; (2) **współczynnik preprocessingu SAM wynosi dokładnie ×0.8** (`1024 / 1280`), jednakowo dla obu serii, niezależnie od docięcia i niezmiennie po obrotach D4; (3) wszystkie kryteria „widoczności po preprocessingu" (blur, przegroda, cienkie ściany) oceniane są przy ×0.8, nie ×0.948; (4) F2 skaluje crop z powrotem do **własnych wymiarów treści danego obrazu**, a nie do jednej stałej pary liczb; (5) zweryfikowano, że utrata ~16% skali roboczej względem pierwotnego założenia **nie podważa decyzji o treningu na całych obrazach** (część III.1, „Rozmiar porów w pikselach roboczych").

**Aktualizacja 2026-08-10 (C) — domknięcie pytań otwartych.** Rozstrzygnięto: `AS1`/`AS1A` i `VAB1`/`VAB11` to **odrębne formulacje** (część III.3); luki w numeracji formulacji nie oznaczają zgubionych danych (część III.1); kolejność E2 przed E3 zostaje utrzymana (część XI); nazwa kolumny rodziny to `material`. Sekcja „do uzupełnienia" nie zawiera już decyzji do podjęcia — wyłącznie wielkości z zamrożoną procedurą wyznaczenia (część XVI.2).

---

# CZĘŚĆ I. PODSTAWY POJĘCIOWE

## I.1. Segment Anything Model (SAM) i Micro-SAM

SAM to duży model fundamentowy do segmentacji obrazów, złożony z trzech części:

1. **Enkoder obrazu** (Vision Transformer, ViT) — przekształca obraz w reprezentację wektorową (krawędzie, kształty, tekstury). Zawiera ponad 99% parametrów modelu.
2. **Enkoder promptów** — przetwarza podpowiedzi użytkownika (punkty, ramki, tekst) na osadzenia.
3. **Dekoder masek** — łączy cechy obrazu z promptem i generuje maskę.

**Micro-SAM** to wariant SAM przystosowany do mikroskopii, z dodatkowym **dekoderem instancyjnym** trenowanym razem z modelem (patrz I.4).

## I.2. LoRA (Low-Rank Adaptation)

Technika oszczędnego dostrajania. Zamiast modyfikować oryginalną macierz wag `W`, zamraża się ją i dodaje niskorangową poprawkę:

```
W' = W + ΔW = W + B·A
```

gdzie `W ∈ ℝ^(d×k)`, `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, ranga `r ≪ min(d, k)`. Liczba trenowanych parametrów spada z `d×k` do `(d+k)×r`. Założenie: adaptacja do nowej domeny zachodzi w niskowymiarowej podprzestrzeni. Tu LoRA aplikowana jest do enkodera obrazu; reszta modelu zamrożona, poza trenowanym dekoderem AIS.

## I.3. Segmentacja instancyjna a semantyczna

- **Semantyczna** — klasa per piksel (por/tło), bez rozróżnienia sąsiednich porów.
- **Instancyjna** — dodatkowo osobne ID każdego pora. To ona jest przedmiotem zadania, bo statystyki materiałowe wymagają rozróżnienia pojedynczych porów.

Maska instancyjna to obraz etykiet: tło = `0`, każdy por = unikalna dodatnia liczba całkowita.

## I.4. AIS kontra AMG — dwa tryby automatycznej segmentacji

- **AMG (Automatic Mask Generation)** — oryginalny tryb SAM: gęsta siatka punktów-promptów, maska per punkt, filtrowanie tysięcy masek. Wolny, wymaga strojenia siatki, słaby na obiektach stykających się granicami (gęste pory).
- **AIS (Automatic Instance Segmentation)** — tryb Micro-SAM: dodatkowy dekoder konwolucyjny (trenowany razem z LoRA) przewiduje pierwszy plan i transformaty odległościowe, instancje powstają przez seeded watershed. Jedna szybka inferencja, lepsza separacja gęstych porów.

**Decyzja:** stosujemy **AIS** (`with_segmentation_decoder=True`). Pipeline i tak generuje targety odległościowe (`PerObjectDistanceTransform`). AMG może posłużyć wyłącznie do jednorazowej diagnostyki końcowej i nie uczestniczy w wyborze modelu ani polityki.

## I.5. Augmentacja danych

Generowanie dodatkowych, etykietowo poprawnych wariantów obrazów treningowych. Augmentacja **online** = w locie podczas treningu (bez zapisu na dysk). Typy:

- **geometryczne** — zmieniają położenie/kształt pikseli; synchronicznie na obrazie i masce;
- **fotometryczne** — zmieniają tylko jasność; maska nietknięta;
- **mask-aware** — modyfikują obraz na podstawie maski albo modyfikują samą maskę.

## I.6. Metryki (definicje robocze)

- **IoU** dwóch masek: `|przecięcie| / |suma|`.
- **Mask IoU** — na pikselach maski (uwzględnia kształt).
- **Bounding box IoU** — na prostokątach otaczających; dla porów wydłużonych/ukośnych zawyża obiekt, dlatego **nie** jest metryką główną.
- **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **F1** = średnia harmoniczna.
- **TP/FP/FN** instancyjnie: predykcja poprawnie dopasowana do GT = TP; predykcja bez dopasowania = FP; instancja GT bez dopasowania = FN.
- **Dopasowanie węgierskie** (`scipy.optimize.linear_sum_assignment`) — optymalne, jednoznaczne przyporządkowanie predykcji do GT.
- **Boundary F1** — F1 na konturach instancji z zadaną tolerancją odległości.
- **Ground truth (GT)** — maska referencyjna (ręczna anotacja eksperta).

## I.7. Pojęcia sterujące eksperymentem

- **Krok optymalizatora** — jedna aktualizacja wag; podstawowa jednostka porównania budżetu.
- **Epoka** — jedno przejście przez zbiór treningowy; raportowana pomocniczo.
- **`T_full`** — pełny budżet treningowy w krokach.
- **Δ_sig** — najmniejsza różnica metryki uznawana za realną (nie szum).
- **Seed** — ziarno RNG; odtwarzalność i szacowanie wariancji.
- **Early stopping** — zatrzymanie, gdy metryka walidacyjna przestaje rosnąć przez `patience` ewaluacji.
- **Screening** — szybkie porównanie kandydata na skróconym budżecie.
- **Ablacja** — usunięcie składnika z pełnej polityki, by zmierzyć jego wkład.

---

# CZĘŚĆ II. ZAMROŻONE ZAŁOŻENIA

Między porównywanymi runami zmienia się **wyłącznie** konfiguracja augmentacji. Jeżeli zmieni się cokolwiek innego, różnicy wyniku nie można przypisać augmentacji.

## II.1. Konfiguracja modelu i treningu

| Element | Wartość | Uzasadnienie |
|---|---|---|
| Commit PEFT-SAM / Micro-SAM | ustalony, potem zamrożony | odtwarzalność |
| Backbone | **ViT-L** (rekomendacja) | kompromis szybkości i jakości; opcjonalny finał ViT-H (XII.4) |
| Bazowy checkpoint | natywny SAM albo generalista EM (II.4) | punkt startowy dostrajania |
| Tryb inferencji | **AIS** (`with_segmentation_decoder=True`) | I.4 |
| Konfiguracja LoRA | domyślna peft-sam, zamrożona | zmiana zmieniałaby zdolność modelu |
| Trenowane części | LoRA enkodera + dekoder AIS | reszta zamrożona |
| `batch_size` | **1** | II.2 |
| Gradient accumulation | **1** (opcjonalnie 2–4, zamrożone od startu) | II.2 |
| `n_objects_per_batch` | **25** (domyślne) | przy setkach porów to on definiuje „szerokość" kroku |
| Mixed precision | domyślna peft-sam, zamrożona | numeryka |
| Funkcja kosztu | domyślna Micro-SAM dla AIS (Dice + BCE + regresja transformat) | II.1 niżej |
| Optymalizator / scheduler / LR | domyślne peft-sam dla batch 1 | brak potrzeby przestrajania |
| Rozdzielczość źródłowa | pliki 1280 × 960; **treść** 1280 × 960 (AS) i 1280 × 890 (K, VAB) po docięciu paska | III.1, IV.1 |
| Docięcie paska informacyjnego | zamrożona stała per mikroskop, stosowana w dataloaderze, synchronicznie na obrazie i masce | IV.1 |
| Preprocessing modelu | dłuższy bok (1280) → 1024 + padding do 1024², **×0.8** | interfejs SAM |
| Postprocessing (watershed) | zamrożony po kalibracji na TRAIN | wpływa na liczbę i separację instancji |
| Podział danych | stały (część III) | inaczej porównania niesparowane |
| Metryki, reguła wyboru checkpointu | stałe (części VI, VIII) | inaczej niespójne kryteria |

**Funkcja kosztu.** W trybie AIS łączy się Dice (globalne pokrycie), BCE (lokalna poprawność piksela) i koszt regresji transformat odległościowych dekodera. Wagi domyślne dla AIS, zamrożone.

## II.2. Uzasadnienie batch_size = 1 i AIS

**Batch size** wpływa na pamięć GPU, szum gradientu, sprzężenie z LR, przepustowość i obsługę kształtów. Wybieramy 1, bo: (1) ViT na 1024² z LoRA jest pamięciożerny; (2) **dwa niezależne źródła niejednorodności kształtu** — obroty D4 zamieniają boki (1280×960 ↔ 960×1280), a treść różni się między seriami (AS 1280×960, K i VAB 1280×890) — przy batchu 1 nie trzeba grupować po kształcie; (3) sygnał gradientu na krok pochodzi z ~25 instancji (`n_objects_per_batch`), więc statystyczne uśrednienie już istnieje; (4) domyślne peft-sam i praca referencyjna używają batch 1. Gdyby krzywe kosztu były zbyt szumiące — gradient accumulation 2–4, ustalone przed startem, niezmieniane w trakcie. **AIS zamiast AMG** — uzasadnienie w I.4.

**Dlaczego kształt wejścia w ogóle ma znaczenie przy batchowaniu.** Enkoder SAM zawsze dostaje płótno 1024 × 1024, więc po paddingu tensor obrazu ma jednolity rozmiar niezależnie od proporcji wejścia. Wąskim gardłem są **etykiety**: maska instancyjna i targety pochodne dekodera AIS (`PerObjectDistanceTransform`) powstają w rozdzielczości sprzed paddingu i zachowują kształt treści. Złożenie ich w jeden tensor batcha wymagałoby identycznych wymiarów albo grupowania próbek po kształcie. Przy `batch_size = 1` pytanie nie powstaje.

**Padding jako koszt, nie jako zagrożenie porównywalności.** Po ×0.8 treść zajmuje 1024 × 768 dla AS (25% płótna to padding) i 1024 × 712 dla K i VAB (30.5% paddingu). To realny narzut obliczeniowy — ViT liczy uwagę także przez piksele bez treści — ale **stały dla danej serii i identyczny we wszystkich runach** (B0, kandydaci, FULL, ablacje, TEST). Nie wpływa więc na atrybucję augmentacji. Różnica udziału paddingu **między seriami** dokłada się natomiast do splotu „rodzina = mikroskop" (III.1) i przy interpretacji przekrojów per rodzina musi być wymieniona obok mikroskopu jako kolejny czynnik, którego nie wolno nazwać efektem materiałowym.

## II.3. Świeży start każdego niezależnego runu

Każdy run porównawczy: wczytuje ten sam bazowy checkpoint; tworzy świeże LoRA i świeży dekoder AIS; nowy optymalizator i scheduler; start od kroku 0. **Zakaz** startu z checkpointu wytrenowanego inną polityką augmentacji. Dopuszczalne wyłącznie **wznowienie tego samego runu** (mechanizm promocji ze screeningu, część IX).

## II.4. Opcjonalny pilotaż wyboru bazowego checkpointu (P0)

Dwa krótkie runy (~30% `T_full`, jeden seed): (a) natywny SAM, (b) generalista EM z Micro-SAM (np. `vit_l_em_organelles` — transfer na SEM materiałowe niepewny, ale tani do sprawdzenia). Zwycięzcę zamraża się jako bazowy checkpoint. Jeżeli P0 pominięty → natywny SAM.

### Uwaga o kolejności (decyzja 2026-08-25)

Zapis „etap F" sugeruje, że P0 wyprzedza G. Dosłownie wzięty jest niewykonalny, bo zależności tworzą cykl:

- **F wymaga G** — żeby cokolwiek wytrenować, potrzebny jest dataloader, preprocessing, LoRA i dekoder AIS, czyli zawartość G;
- **F wymaga L** — „~30% `T_full`" odwołuje się do wielkości wyznaczanej dopiero w kroku L;
- **L wymaga F** — E0 musi wystartować z jakiegoś bazowego checkpointu.

**Rozstrzygnięcie: litery w części XVII są kolejnością zamrażania decyzji, nie kolejnością pisania kodu.** G nazywa się „zamrożenie", a nie „budowa"; stos powstaje wcześniej niezależnie od liter, a F stoi przed G dlatego, że bazowy checkpoint jest jedną z wartości, które G zamraża. Kolejność wykonawcza brzmi: zbudować stos → P0 → zamrozić stos wraz ze zwycięzcą → L.

Konsekwencje zamrożone razem z tą decyzją:

1. **Budżet P0 wyrażamy w krokach bezwzględnych, nie jako ułamek `T_full`.** P0 nie uczestniczy w atrybucji, więc jego budżet nie musi być współmierny z niczym; zapis „~30%" czytamy jako szacunek rzędu wielkości.
2. **P0 jest jednocześnie pierwszym testem dymnym pipeline'u.** Pierwszy udany run trzeba wykonać tak czy inaczej, żeby sprawdzić, że docięcie nie psuje masek, że AIS się uczy i że koszt spada. Zrobienie z niego P0 kosztuje dodatkowo tylko drugi run.
3. **P0 porównuje checkpointy bez augmentacji.** Checkpoint wygrywający w B0 nie musi wygrywać pod FULL; sprawdzenie tego byłoby macierzą 2×2 poza budżetem. Ryzyko przyjęte świadomie i odnotowane przy wyniku.
4. **Reguła awaryjna pozostaje w mocy.** Jeżeli instalacja Micro-SAM albo stos sprawią kłopot, P0 odpada i obowiązuje natywny SAM — bez blokowania dalszych kroków.

Stan środowiska na 2026-08-25: `micro_sam` ani `peft_sam` nie są zainstalowane w żadnym venvie, a w cache Micro-SAM są wyłącznie `vit_b` i `vit_h` — brak ViT-L, który plan zamraża jako backbone (II.1). Nazwę `vit_l_em_organelles` należy zweryfikować w rejestrze modeli przy instalacji, a nie przyjmować za pewnik.

---

# CZĘŚĆ III. DANE I PODZIAŁ

## III.1. Struktura danych (wynik inwentaryzacji)

Zbiór zawiera **707 obrazów w 31 formulacjach**. Każda formulacja pochodzi z jednej syntezy (jedna próbka, pocięta na kostki i sfotografowana wielokrotnie). **Nie śledzimy kostek** — najdrobniejszą jednostką grupowania jest formulacja.

**Rodzina pianki jest identyfikowalna** z prefiksu nazwy formulacji, więc wchodzi do stratyfikacji i do przekrojów raportowania:

| Rodzina | Formulacje | Obrazy | Udział | Mikroskop |
|---|---:|---:|---:|---|
| AS | 20 | 599 | 84.7% | M1 |
| K | 5 | 69 | 9.8% | M2 |
| VAB | 6 | 39 | 5.5% | M2 |
| **Razem** | **31** | **707** | **100%** | |

### Dwa mikroskopy

Zdjęcia powstały na dwóch mikroskopach o różnej kalibracji. Iloczyn `pixel_size_um × powiększenie` jest stały dla każdego z nich: **129.61 dla AS (M1)** oraz **99.22 dla K i VAB (M2)**. Stosunek 1.306 oznacza, że przy tym samym nominalnym powiększeniu oba mikroskopy dają inną skalę fizyczną.

**Wniosek wiążący: nominalne powiększenie nie jest miarą skali i zostaje wycofane** jako wymiar stratyfikacji i raportowania. AS@40× ma 3.24 µm/px, a VAB@40× — 2.48 µm/px (30% różnicy); z drugiej strony K@30× (3.307) i AS@40× (3.240) różnią się o 2%, czyli są praktycznie tą samą skalą. Powiększenie pozostaje w manifeście wyłącznie jako metadana pochodzenia.

**Rodzina jest skonfundowana z mikroskopem** (AS = M1, K i VAB = M2). Oznacza to, że przekroje „per rodzina" są jednocześnie przekrojami „per mikroskop" — różnic nie wolno interpretować jako czysto materiałowych. Dla podziału danych to zaostrzenie wymagań, nie problem (III.4).

### Skale występujące w zbiorze

| pixel_size_um | Obrazy | Źródło | scale_bin |
|---:|---:|---|---|
| 3.307292 | 71 | K@30× (68) + VAB1@30× (3) | coarse |
| 3.24023 | 554 | AS@40× | coarse |
| 2.59219 | 43 | AS@50× | fine |
| 2.480469 | 33 | VAB@40× | fine |
| 0.7632211 | 1 | K1@130× | outlier |
| 0.4960938 | 2 | VAB11@200× | outlier |
| 0.2480469 | 1 | VAB3@400× | outlier |
| 0.25922 | 2 | AS4@500× | outlier |

Skale nie tworzą kontinuum, tylko **dwa klastry** rozdzielone czynnikiem ~1.3, plus sześć odstających zbliżeń:

```text
coarse:  3.24–3.31 µm/px → 625 obrazów (88.4%)   rozrzut wewnętrzny 2%
fine:    2.48–2.59 µm/px →  76 obrazów (10.7%)   rozrzut wewnętrzny 4.5%
outlier: 0.25–0.76 µm/px →   6 obrazów  (0.8%)   3–13× drobniejsze
```

Interpretacja: w binie `coarse` ten sam por zajmuje najmniej pikseli; w `fine` zajmowałby o ~30% pikseli więcej w każdym wymiarze. Cała rzeczywista zmienność skali w zbiorze to **jeden skok o czynnik 1.3**.

### Rozkład powiększenia 50× i zasilanie binu `fine`

Powiększenie 50× występuje w **20 formulacjach — wszystkich z rodziny AS** — po 2–3 obrazy, razem 43. W K i VAB nie występuje wcale. Bin `fine` ma więc dwa niezależne źródła: **AS@50× (43 obrazy) i VAB@40× (33 obrazy)**. Każda formulacja AS zawiera oba biny skali; każda formulacja VAB zawiera bin `fine`; formulacje K zawierają wyłącznie `coarse`.

### Obrazy odstające (`scale_outlier`)

Sześć zbliżeń przy 130–500× to wizualnie inne zadanie obrazowe. Decyzja: **pozostają w TRAIN z `q = 1.00`, są wyłączone z VALIDATION, TEST, z kalibracji `q` i z metryk per `scale_bin`**, i są jawnie raportowane jako nieoceniane. Powód wyłączenia z kalibracji: policzone razem z nimi `q_max` wynosi 13.33 zamiast 1.33.

### Luki w numeracji formulacji

W numeracji brakuje niektórych oznaczeń (m.in. `K4`, `AS2`, `AS8`–`AS14`). **Decyzja: to luki nazewnictwa, nie zgubione dane.** Populacją eksperymentu jest zbiór zaanotowany, a jego jedynym autorytatywnym rejestrem jest eksport z Label Studio, z którego budowany jest manifest. Formulacja, która nie występuje w eksporcie, nie została zaanotowana i nie wchodzi do eksperymentu — niezależnie od tego, czy jej numer sugeruje istnienie próbki. Nie prowadzi się osobnego śledzenia „brakujących numerów"; liczbą wiążącą jest 31 formulacji i 707 obrazów.

### Geometria obrazów i pasek informacyjny

Weryfikacja wymiarów (2026-08-10) obaliła pierwotne założenie planu o rozdzielczości źródłowej 960 × 1080. Stan faktyczny:

| Seria | Mikroskop | Plik na dysku | Pasek informacyjny | **Treść** |
|---|---|---|---|---|
| AS | M1 (TM3000) | 1280 × 960 | usunięty przed anotacją (akwizycja 1280 × 1040, ścięte 80 wierszy) | 1280 × 960 |
| K, VAB | M2 (SU8000) | 1280 × 960 | **obecny**, dolne 70 wierszy (`y ∈ [890, 959]`) | 1280 × 890 |

Potwierdzenie niezależne od oglądania pikseli daje kolumna manifestu `panel_cropped_px`, liczona jako `sidecar.DataSize_h − wysokość_pliku`: dla AS wynosi 80 (panel już ścięty), dla K i VAB — 0 (panel wciąż w pliku).

**Współczynnik preprocessingu SAM wynosi dokładnie ×0.8.** SAM skaluje **dłuższy bok** do 1024 i dopełnia do kwadratu 1024 × 1024. Dłuższym bokiem jest szerokość 1280, więc `1024 / 1280 = 0.8`. Trzy konsekwencje warte odnotowania, bo upraszczają cały pipeline:

- współczynnik jest **jeden dla całego zbioru** — pierwotne 0.948 wynikało z błędnego założenia, że dłuższym bokiem jest 1080;
- pasek leży na **wysokości**, a skalowanie ustala **szerokość**, więc decyzja o docięciu **nie zmienia współczynnika**;
- po obrotach D4 obraz ma 960 × 1280 (albo 890 × 1280) — dłuższym bokiem nadal jest 1280, więc ×0.8 przeżywa augmentację geometryczną.

Płótno robocze: **AS → 1024 × 768** (25% paddingu), **K i VAB → 1024 × 712** (30.5% paddingu). Interpretacja narzutu paddingu — II.2.

**Dlaczego pasek musi zostać docięty, a nie zostawiony.** Pasek występuje wyłącznie w K i VAB, a te są idealnie skonfundowane z mikroskopem M2 (i z rodziną). Zostawiony byłby artefaktem doskonale skorelowanym z rodziną materiału: każda różnica w przekroju „per rodzina", zaraportowana potem jako efekt materiałowy albo aparaturowy, byłaby nierozróżnialna od „model nauczył się czarnego paska". Przy liczebnościach przekrojów rzędu kilkudziesięciu obrazów to wystarcza, by unieważnić wnioski. Dodatkowo pasek to obszar bez porów, więc zaburza gęstość instancji i targety transformat odległościowych dekodera AIS. Reguła docięcia — IV.1.

### Rozmiar porów w pikselach roboczych

Utrata ~16% skali roboczej względem pierwotnego założenia (×0.8 zamiast ×0.948) podnosi pytanie, czy najmniejsze anotowane pory pozostają w zasięgu modelu przy treningu na całych obrazach. Zmierzono z manifestu:

| Wielkość | Źródłowo [px] | Roboczo (×0.8) [px] |
|---|---:|---:|
| Najmniejsza średnica równoważna w całym zbiorze (`pore_equivalent_diameter_min_px`, minimum) | 6.86 | **5.49** |
| Najmniejsza mediana per obraz (`pore_equivalent_diameter_median_px`, minimum) | 58.66 | **46.93** |

Najmniejszy por w całym zbiorze ma roboczo ~24 px² powierzchni, a typowy por nawet w najtrudniejszym obrazie — kilkadziesiąt pikseli średnicy. **Wniosek: decyzja o treningu na całych obrazach, bez wycinków, pozostaje bezpieczna** i nie wymaga rewizji. Warunek uznaje się za **sprawdzony i zamknięty** — nie jest otwartą bramką. Zapis jest celowy: pokazuje, że różnica względem pierwotnego założenia została zmierzona, a nie przeoczona.

Dwie konsekwencje pochodne, obie łagodne:

- **Blur (F3b) jest bezpieczny dla porów.** Augmentacja działa w rozdzielczości źródłowej, więc `sigma ≤ 0.8` px źródłowych daje po skalowaniu efektywnie ≤ 0.64 px roboczych — por o średnicy 5.5 px to przeżywa.
- **Boundary F1 przy tolerancji 2 px roboczych zachowuje się różnie w zależności od rozmiaru pora** — dla mediany 47 px to 4% średnicy, dla ekstremum 5.5 px aż 36%. Próg zostaje bez zmian, ale interpretacja jest odnotowana w VI.2.

**Czego te liczby nie mówią.** Manifest zna średnice porów, ale **nie zna grubości ścian**. To ściany, a nie pory, są tym, czemu blur i skalowanie ×0.8 realnie zagrażają. Ocena grubości ścian po pełnym preprocessingu należy do Fazy 0 (część VII) i nie da się jej rozstrzygnąć z manifestu.

## III.2. Manifest metadanych (automatyczny, lekki)

Ciężki, ręczny rekord per obraz jest zbędny. Nazwy plików kodują formulację i powiększenie, a `pixel_size_um` jest znany per obraz — wystarczy **jeden automatycznie generowany manifest** (np. CSV/DataFrame), jeden wiersz na obraz. Wszystkie kolumny pochodne wyprowadza się deterministycznie, bez pracy ręcznej:

```text
image_id           # z nazwy pliku
formulation        # sparsowane z nazwy pliku; jednostka grupowania splitu
material           # AS | K | VAB — prefiks formulacji (rodzina pianki)
microscope         # M1 | M2 — wyprowadzone z instrumentu SEM (łańcuch niżej)
microscope_source  # sem_sidecar | series_map | pixel_size_product
magnification      # metadana pochodzenia; NIE używana do stratyfikacji
                   #   ani do kalibracji skali
pixel_size_um      # rzeczywista skala per obraz [µm/px]
scale_bin          # coarse | fine | outlier — wyprowadzone z pixel_size_um
scale_outlier      # bool — (scale_bin == outlier)
q_max_i            # pixel_size_um / 2.480469; polityka zamrożona w V.2
load_crop_bbox     # zamrożona ramka treści per mikroskop (IV.1)
content_bbox       # wynik detektora paska; diagnostyka i asercja, NIE
                   #   używany bezpośrednio przez dataloader
panel_cropped_px   # ile wierszy panelu usunięto już z pliku źródłowego
source_path
mask_path
```

Reguły wyprowadzania (zamrożone):

```text
microscope — łańcuch źródeł, pierwsze trafienie wygrywa:
  1. instrument z sidecara SEM:  TM3000 -> M1,  SU8000 -> M2
  2. mapa per seria (gdy brak sidecara)
  3. iloczyn pixel_size_um * magnification: ~129.61 -> M1, ~99.22 -> M2
Wybrany poziom zapisywany w microscope_source.

scale_bin  = coarse   jeżeli pixel_size_um >= 3.0
           = fine     jeżeli 2.4 <= pixel_size_um < 3.0
           = outlier  jeżeli pixel_size_um < 2.4
scale_outlier = (scale_bin == "outlier")

load_crop_bbox = (0, 0, 1280, 960)  dla microscope == M1
               = (0, 0, 1280, 890)  dla microscope == M2
```

**Dlaczego `microscope` z sidecara, a nie z iloczynu.** Iloczyn `pixel_size_um × magnification` jest kruchy: `magnification` bywa parsowane z nazwy pliku tylko dla AS, a dla K i VAB pochodzi z sidecara — gdy powiększenia zabraknie, reguła nie daje wyniku. Instrument SEM jest odczytem bezpośrednim i nie zależy od nazewnictwa plików. Iloczyn zostaje jako **niezależny test spójności**: gdy jest policzalny, a jego wynik przeczy przypisanemu `microscope`, powstaje ostrzeżenie do obejrzenia — bez zmiany wartości kolumny.

**Dlaczego `scale_bin` regułą bezwzględną.** Wcześniejsza implementacja wyznaczała odstępstwo skali **względnie**, jako odchylenie od mediany serii. Na obecnych danych obie reguły dają ten sam zbiór sześciu obrazów, ale zgodność jest przypadkowa, nie strukturalna. Ponieważ progi uczestniczą w kalibracji `q` i w stratyfikacji, autorytatywna jest **reguła bezwzględna** z zamrożonymi progami 3.0 i 2.4. Reguła względna zostaje wyłącznie jako diagnostyka wykrywająca **nowe** dane, które nie mieszczą się w zamrożonych binach.

**Znaczenie kolumn skali.** `pixel_size_um` mówi, ile mikrometrów pianki przypada na jeden piksel — czyli jak duży jest por *w pikselach*, a to jest jedyna wielkość, którą model faktycznie widzi. `scale_bin` grupuje obrazy w dwa naturalne klastry skali (III.1) i pełni trzy funkcje: ustala zakres `q` dla F2, pilnuje reprezentacji obu skal w każdym zbiorze i pozwala raportować wyniki osobno dla każdej skali (bez tego efekt F2 tonie w średniej zdominowanej przez 88% obrazów `coarse`). `scale_outlier` trzyma sześć zbliżeń poza kalibracją i poza oceną.

**Kalibracja skali.** `pixel_size_um` jest **rzeczywisty per obraz**, nie nominalny per powiększenie. Rozkład nie degeneruje się do dwóch wartości — ma osiem wartości układających się w dwa klastry plus outliery, a `q` kalibruje się per obraz (V.2).

Nie tworzy się ręcznych tagów diagnostycznych (gęste pory, cienie, cienkie ściany itp.) — są zbędne. Wszystkie przekroje raportowania wynikają wprost z manifestu: **per formulacja**, **per `scale_bin`**, **per `material`** i **per mikroskop**.

**Wersjonowanie manifestu.** Każda zmiana zestawu kolumn podnosi wersję manifestu. Do metadanych runu zapisuje się wielkości, które czynią manifest samoopisującym się: progi binów skali (3.0 i 2.4), stałą referencyjną `q` (2.480469), mapę instrumentów na mikroskopy, mapę `load_crop_bbox` oraz wersję reguł wyprowadzania.

## III.3. Zasada grupowania: formulacja jako jednostka splitu

**Przeciek (data leakage).** Obrazy tej samej formulacji pochodzą z jednej syntezy i są silnie skorelowane. Gdyby część trafiła do treningu, a część do walidacji/testu, model „widziałby" w ocenie materiał praktycznie znany z treningu — metryki byłyby zawyżone.

**Zasada:** ponieważ nie śledzimy kostek, najdrobniejszą bezpieczną jednostką grupowania jest formulacja. **Wszystkie obrazy jednej formulacji trafiają w całości do tego samego zbioru.**

**Formulacje o wspólnym prefiksie — rozstrzygnięte.** Pary `AS1` / `AS1A` oraz `VAB1` / `VAB11` budziły wątpliwość, czy nie są wariantami tej samej syntezy (co wymuszałoby wspólny zbiór). **Decyzja: to odrębne formulacje i odrębne jednostki splitu.** Podstawa: wiedza materiałowa o pochodzeniu próbek, potwierdzona brakiem współdzielonych skrótów plików (`duplicate_file_hash`) w obrębie tych par. Zapis jest świadomie ostrożny — brak duplikatu skrótu dowodzi wyłącznie, że to różne **pliki**, nie że to różne **syntezy**; wiążąca jest tu decyzja ekspercka, a test automatyczny ją jedynie wspiera.

Konsekwencja praktyczna: **grupa splitu jest tożsama z formulacją** i nie potrzeba mapy nadpisań łączącej formulacje w większe grupy.

## III.4. Architektura oceny (31 formulacji)

Przy kilkunastu–dwudziestu kilku formulacjach mieści się porządny, klasyczny grupowy podział z prawdziwym, nietkniętym zbiorem testowym — nie ma potrzeby stosowania leave-one-formulation-out (który przy tylu formulacjach byłby kosztowny).

```text
FORMULACJE (31) → grupowo, ze stratyfikacją po `material × scale_bin`:
├── TRAIN        ~65–70% formulacji   (≈ 21)
├── VALIDATION   ~15% formulacji      (≈ 5)
└── TEST         ~15–20% formulacji   (≈ 5)   (nietknięty do samego końca)
```

Rola zbiorów:

- **TRAIN** — dostrajanie modelu.
- **VALIDATION** — wybór checkpointu, próg `Δ_sig`, wszystkie decyzje o rodzinach augmentacji (screening, strojenie, ablacje). Pełni funkcję jednolitej linijki względnej.
- **TEST** — otwierany **raz**, po zamrożeniu polityki, do liczb nagłówkowych i do potwierdzenia atrybucji. Nie uczestniczy w żadnej decyzji.

Ponieważ mamy prawdziwy, wydzielony od początku TEST, nie ma „skażenia" oceny końcowej — problem, który wymuszał wcześniej rotacyjne LOFO, tu nie występuje.

### Dobór liczności

Split definiuje się **przez proporcje i stratyfikację**, nie przez sztywną listę formulacji. Zalecenia:

- każdy zbiór dostaje formulacje z obu (wszystkich) rodzajów pianek, jeśli materiał jest identyfikowalny;
- **każdy zbiór musi zawierać ≥1 formulację z mikroskopu M2** (rodzina K albo VAB). Bez tego ocena nie mierzy transferu międzymikroskopowego, który jest realnym ryzykiem tego zbioru (III.1);
- **każdy zbiór musi zawierać oba biny skali.** Warunek spełnia się niemal automatycznie: każda formulacja AS ma obrazy `coarse` (40×) i `fine` (50×), a każda formulacja VAB ma `fine` (40×). Formulacje K są wyłącznie `coarse`, więc sam K nie wystarcza do pokrycia binu `fine`;
- rodzina VAB jest nieliczna (39 obrazów, formulacje po 2–9 obrazów). Jedna formulacja VAB w TEST da ~6–8 obrazów — to dopuszczalne, ale liczebność trzeba **jawnie zaraportować** przy każdym przekroju;
- rodzina AS dominuje (84.7% obrazów). Przy proporcjonalnym samplerze (III.7) K i VAB dostaną razem ~15% kroków treningowych — to świadoma decyzja, monitorowana metrykami per rodzina, a nie usterka do naprawienia oversamplingiem;
- formulacje skrajnie nieliczne (1 obraz) trafiają do TRAIN i są raportowane jako niereprezentowane w ocenie;
- przy 31 formulacjach VALIDATION i TEST po ~5 formulacji są wystarczające i grupowy k-fold (III.6) nie jest konieczny do decyzji o polityce. Pozostaje wskazany do publikacji oraz w przekrojach nielicznych (VAB, bin `fine`), gdzie pojedynczy split daje cienkie liczebności.

## III.5. Procedura tworzenia splitu (deterministyczna, raz)

```text
1. Zbudować manifest (III.2).
2. Zgrupować obrazy po formulacji.
3. Dla każdej formulacji policzyć: liczbę obrazów, `material`, mikroskop,
   obecność binów `coarse` / `fine`, liczbę obrazów `scale_outlier`.
4. Rozdzielić formulacje na TRAIN/VAL/TEST wg proporcji z III.4, zachowując stratyfikację
   po `material × scale_bin`, z ustalonym seedem podziału.
5. Zapisać split jako listę formulacji per zbiór (split_id).
6. Zamrozić split; TEST zamknąć do zakończenia E9.
7. Zaraportować w każdym zbiorze rozkład rodzin, mikroskopów i binów skali, z liczebnościami.
8. Zweryfikować twarde warunki z III.4: ≥1 formulacja M2 w VAL i w TEST; oba biny skali
   obecne w każdym zbiorze. Jeżeli seed podziału ich nie spełnia — odrzucić i losować ponownie.
```

Split jest **stały dla wszystkich runów**. Golden gallery (część VII) korzysta wyłącznie z TRAIN.

### Wynik wykonania (2026-08-25, [WYKONANE])

Split `split_v1`, seed podziału **20260825**, wybrany jako argmin kosztu balansu spośród 50 000 kandydatów (100% z nich spełniało twarde warunki — komplet warunków jest siatką bezpieczeństwa, nie filtrem; pracę wykonuje kwota formulacji). Artefakty: `split_v1.csv` (jeden wiersz na obraz), `split_v1_report.md`, `split_v1_metadata.json` (seed, sha256 manifestu, kwota, ograniczenia, wagi kosztu). Kod: `scripts/create_dataset_split.py`, pakiet `data_prep/split/`.

| zbiór | formulacji | obrazów | do oceny | udział | coarse | fine | outlier | instancji |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 21 | 494 | 488 | 69.6% | 437 | 51 | 6 | 25 290 |
| VALIDATION | 5 | 107 | 107 | 15.3% | 94 | 13 | 0 | 4 900 |
| TEST | 5 | 106 | 106 | 15.1% | 94 | 12 | 0 | 5 656 |

- **VALIDATION:** AS1A, AS20, AS22, K5, VAB5
- **TEST:** AS19, AS24, AS3, K3, VAB6
- **TRAIN:** pozostałe 21 formulacji

Wszystkie twarde warunki z III.4 spełnione: ≥1 formulacja M2 w każdym zbiorze, oba biny skali w każdym zbiorze, ≥8 obrazów `fine` oraz ≥10 obrazów K i ≥5 obrazów VAB w VALIDATION i w TEST. Przekrój `fine` w VALIDATION wynosi 13 obrazów — zgodnie z przewidywaniem z V.2, decyzja o F2 musi opierać się na metrykach per `scale_bin`, nie na metryce globalnej.

**A_min_fragment = 432.0 px²** — P1 z 25 253 instancji TRAIN, mierzony po docięciu do `load_crop_bbox`, z pominięciem obrazów `scale_outlier`. Zero instancji utraconych przez docięcie (instancje sięgają w pasek, ale żadna nie leży w nim w całości). Wartości per bin, jako diagnostyka: `coarse` 451.5, `fine` 217.8 px². Rozjazd wynika z tego, że powierzchnia w pikselach skaluje się z kwadratem stosunku rozdzielczości; wartość globalna jest de facto wartością `coarse`, co jest spójne z zastosowaniem, bo F2 stosuje `q > 1` wyłącznie do binu `coarse`. Gdyby reguła fragmentów trafiła kiedyś na obrazy `fine`, próg 432 byłby o ~2× za ostry.

**Trzy rozstrzygnięcia, których ten plan wcześniej nie zawierał** (podjęte przy wykonaniu, zamrożone):

1. **Obraz `scale_outlier` w formulacji spoza TRAIN jest odrzucany w całości**, a nie przenoszony do TRAIN. Zapis z III.1 („outliery pozostają w TRAIN") nie może wygrać z zasadą grupowania z III.3 — przeniesienie samego obrazu rozbiłoby formulację na dwa zbiory i przywróciło przeciek. Strata jest jawna w kolumnie `used` pliku splitu. W `split_v1` kosztowało to **zero obrazów**: wszystkie cztery formulacje z outlierami (AS4, K1, VAB3, VAB11) trafiły do TRAIN.
2. **`VAB11` jest przypięta do TRAIN** przed losowaniem. Ma 4 obrazy, z czego 2 to outliery, więc w VALIDATION lub TEST dałaby przekrój VAB z 2 obrazów. To jedyne twarde przypisanie w całej procedurze.
3. **Wybór splitu = argmin kosztu balansu**, nie pierwszy dopuszczalny kandydat. Koszt jest sumą odchyleń L1 od proporcji docelowej, liczonych na **obrazach i instancjach**, a nie na formulacjach — formulacje mają 4–40 obrazów, więc sama kwota formulacyjna nie kontroluje niczego, co ma znaczenie dla mocy oceny. Koszt nie widzi żadnej metryki ani modelu, więc jego minimalizacja nie może przechylić wyniku eksperymentu — może wyłącznie wyrównać proporcje raportowanych przekrojów.

**Zamknięcie TEST pozostaje do kroku G.** Split jednoznacznie wskazuje zbiór testowy, ale nie istnieje jeszcze kod, który by go bronił; bramka (wczytanie `split == "test"` wymaga jawnej flagi i jest logowane) powstaje razem z dataloaderem.

## III.6. Opcjonalna robustność: grupowy k-fold

Dla finalnej polityki (i jej głównego konkurenta) można — zamiast polegać na pojedynczym VALIDATION — wykonać **grupowy k-fold CV** (np. 5-fold) na zbiorze rozwojowym (TRAIN + VALIDATION), z formulacją jako grupą. Daje to rozkład wyniku zamiast jednego punktu i wzmacnia wiarygodność decyzji o polityce, zwłaszcza przy dolnym końcu X. TEST pozostaje osobny i nietknięty. Grupowy k-fold jest wskazany dla publikacji; przy dużym X i wyraźnych efektach można poprzestać na TEST + seedy.

## III.7. Sampler obrazów treningowych

Losowanie obrazów do batchy **proporcjonalne do liczebności, bez oversamplingu**, zamrożone dla wszystkich runów. Nierównowagę rzadkich podzbiorów (bin `fine` — 10.7%; rodziny K i VAB — 15.3%) monitorują metryki per `scale_bin`, per rodzina i per formulacja; ewentualna zmiana wag musiałaby nastąpić przed startem porównań i obowiązywać wszystkie runy — nigdy w trakcie. Świadoma konsekwencja: model bez augmentacji skali nauczy się głównie skali `coarse`, i to jest właśnie hipoteza, którą testuje F2.

### Wynik wykonania (2026-08-25, [WYKONANE])

Zaimplementowane jako `materials_vision/data/sampling.py` (`ProportionalImageSampler`) plus `materials_vision/data/split_io.py` (`load_split`). Konfiguracja trafia do metadanych runu przez `sampler_run_metadata()`.

**Realna ekspozycja na `split_v1`** — przy `batch_size = 1` udział obrazów jest dokładnie udziałem kroków optymalizatora, więc te liczby raportuje się obok metryk per przekrój:

| przekrój | obrazów TRAIN | udział kroków |
|---|---:|---:|
| AS | 422 | 85.4% |
| K | 45 | 9.1% |
| VAB | 27 | 5.5% |
| `coarse` | 437 | 88.5% |
| `fine` | 51 | 10.3% |
| `outlier` | 6 | 1.2% |

Epoka to **494 kroki**.

**Dwa rozstrzygnięcia, których ten plan wcześniej nie zawierał** (podjęte przy wykonaniu, zamrożone):

1. **Permutacja epoki, nie losowanie ze zwracaniem.** „Proporcjonalne do liczebności" nie odróżnia tych dwóch wariantów. Wybrano permutację: daje udziały dokładnie proporcjonalne w *każdej* epoce, a nie tylko w oczekiwaniu, i pozostawia „epokę" dobrze zdefiniowaną na potrzeby raportowania pomocniczego z I.7. Porównania nadal idą po krokach optymalizatora (V.1).
2. **Sampler ma własny strumień RNG, odseparowany od augmentacji.** To warunek konieczny sparowanych porównań z X.1, a nie detal implementacyjny. `DataLoader(shuffle=True)` czerpie permutację z globalnego generatora torcha, którego stan zależy od tego, ile losowości zużyło wszystko inne — a to różni się między politykami augmentacji. B0 i FULL przy tym samym seedzie dostałyby wtedy **różną kolejność obrazów** i część zmierzonej różnicy byłaby szumem kolejności udającym efekt augmentacji. Permutacja jest więc zasiewana z lokalnego generatora, wyłącznie z pary `(run_seed, epoch)` przez `blake2b`. Zależność musi być jednostronna: augmentacja nie może wpływać na kolejność obrazów, odwrotnie może. Broni tego test `test_image_order_is_immune_to_augmentation_randomness`.

**Bramka na TEST została domknięta tutaj, nie w kroku G.** Strażnikiem musi być ten, kto czyta plik splitu, a jest nim `load_split`: odczyt `subset="test"` bez `allow_test=True` podnosi `LockedTestSetError`, a odczyt z flagą zostawia wpis WARNING w logu. `load_split` opcjonalnie weryfikuje też `sha256` manifestu wobec metadanych splitu, więc split nie może po cichu przeżyć manifestu, z którego powstał.

---

# CZĘŚĆ IV. PREPROCESSING I KOLEJNOŚĆ PIPELINE

## IV.1. Deterministyczny preprocessing

Zapisać jednoznacznie: źródłowy `dtype`, zakres intensywności, moment konwersji do `float32`, clipping po fotometrii, normalizację, wartości paddingu, interfejs kanałów.

**Usuwanie elementów nieobrazowych — reguła zamrożona.** Dolny pasek informacyjny SEM usuwany jest **deterministycznie, przed augmentacją**. To preprocessing, nie augmentacja.

```text
load_crop_bbox:
  microscope = M1 (AS)      -> (0, 0, 1280, 960)   # pasek już ścięty w pliku
  microscope = M2 (K, VAB)  -> (0, 0, 1280, 890)   # 70 wierszy do odcięcia
```

Cztery reguły towarzyszące:

1. **Stała jest autorytatywna, detektor nie uczestniczy w ścieżce treningowej.** Dataloader czyta wyłącznie zamrożoną wartość `load_crop_bbox`; detektor paska nie jest wywoływany podczas treningu (koszt zerowy). Jego wynik, kolumna `content_bbox`, służy do **jednorazowej asercji przy budowie manifestu** — jedno porównanie po wszystkich wierszach, wykonywane raz. Rola asercji jest wyłącznie zabezpieczająca: gdy dojdą nowe dane z tego samego mikroskopu, jest to jedyny mechanizm, który wykryje zmianę wysokości paska przez oprogramowanie akwizycji.
2. **Kluczowanie po `microscope`, nie po serii ani katalogu.** Pasek jest właściwością aparatury, więc zmienna przyczynowa to mikroskop. Mapa mieszka w konfiguracji i trafia do metadanych runu.
3. **Docięcie w dataloaderze, nie na dysku.** Pliki źródłowe pozostają nietknięte, więc `file_hash` w manifeście nadal opisuje to, co faktycznie leży na dysku, a reguła docięcia jest wersjonowana razem z konfiguracją. Materializacja dociętych kopii na dysku tworzyłaby drugi zbiór wymagający osobnego wersjonowania i skrótów — bez korzyści metodologicznej.
4. **Docięcie jest częścią preprocessingu inferencji.** Obrazy trafiające do modelu wdrożeniowego mają pasek. Gdyby docięcie żyło wyłącznie w przygotowaniu zbioru treningowego, powstałby rozjazd trening/produkcja.

**Maski przy docięciu.** Poligony przesuwa się o `(x0, y0)` ramki i przycina do jej granic. Anotacja wykraczająca poza `y = 889` jest **cięta prostą wzdłuż krawędzi docięcia** — nie odrzucana. Pozostały fragment zachowuje się, jeżeli jest spójny i ma powierzchnię ≥ `A_min_fragment` (V.2); instancja dotykająca krawędzi docięcia dostaje flagę `border_instance` i podlega tym samym regułom co instancje przecięte krawędzią cropu, w tym wykluczeniu z metryk morfologicznych spójnie po stronie GT i predykcji. Po cięciu — gęsta renumeracja ID.

**Doprecyzowanie zakresu `A_min_fragment` (2026-08-25, przy implementacji).** Próg dotyczy **wyłącznie instancji, którym docięcie realnie zabrało powierzchnię**. Instancja leżąca w całości wewnątrz ramki treści jest prawdziwą anotacją i przeżywa niezależnie od rozmiaru. Powód jest arytmetyczny: `A_min_fragment` to **P1 rozkładu powierzchni instancji GT**, więc zastosowany do instancji nietkniętych kasowałby z definicji dolny percentyl prawdziwej anotacji — na każdym obrazie, także na 599 obrazach AS, gdzie docięcia nie ma w ogóle. Pierwsza implementacja stosowała próg globalnie i usuwała 404 instancje, z czego 211 na M1; po zawężeniu zakresu strata na M1 wynosi zero. To samo zawężenie dotyczy warunku spójności („zachowuje się, jeżeli jest spójny"): instancja rozspójniona przez nakładanie się sąsiada, a nie przez cięcie, zachowuje wszystkie swoje piksele pod jednym ID.

**Zmierzone zachowanie docięcia na całym zbiorze (2026-08-25, [WYKONANE]).** Implementacja: `materials_vision/data/instances.py`, `apply_content_crop`.

| | M1 (bez docięcia) | M2 (pasek 70 wierszy) |
|---|---:|---:|
| instancji na wejściu | 31 130 | 4 716 |
| instancji po docięciu | **31 130** | 4 709 |
| przeciętych przez krawędź | 0 | 399 |
| odrzuconych poniżej `A_min_fragment` | 0 | 7 |
| rozspójnionych przez cięcie | 0 | 0 |

Trzy wnioski: (1) seria AS przechodzi przez docięcie **bez żadnej straty**, co jest warunkiem zgodności z liczbami instancji w manifeście; (2) 399 przeciętych instancji zgadza się z 404 z weryfikacji C5 niżej — tamta liczba idzie po wierzchołkach poligonów, ta po realnej utracie powierzchni; (3) reguła „zachowaj największy spójny fragment" **nie uruchomiła się ani razu** na prawdziwych danych, więc pozostaje siatką bezpieczeństwa, a nie mechanizmem roboczym. Łączny koszt docięcia to 7 instancji w całym zbiorze, czyli 0.02%.

**Weryfikacja empiryczna (C5, [WYKONANE]).** Sprawdzono maksymalną współrzędną `y` wierzchołków poligonów dla całej serii K/VAB (manifest v2, kolumna `n_instances_below_crop_bbox`, licząca instancje z co najmniej jednym wierzchołkiem na `y ≥ 890`). Wynik: **108 z 108 obrazów M2 (100%)** ma co najmniej jedną taką instancję, łącznie **404 instancje** w całym zbiorze, średnio ~3.7 na obraz tam, gdzie występują. Zgadza się to z prostym rachunkiem prawdopodobieństwa dla gęstości porów w tej serii (~44 instancji/obraz, mediana średnicy ~127 px, treść 890 px wysokości) — nie jest artefaktem pojedynczych obrazów. **Wniosek: reguła cięcia poligonów opisana wyżej nie jest zabezpieczeniem na rzadki przypadek brzegowy — jest obowiązkowym elementem pipeline'u dla całej podserii K/VAB, bez którego samo docięcie paska systematycznie okalecza adnotacje w niemal każdym obrazie tej podserii.**

**Kanały.** Obrazy monochromatyczne (trzy identyczne kanały). Pipeline pracuje na **jednym kanale roboczym**; powielenie do RGB dopiero na końcu. Cała augmentacja fotometryczna działa na jednym kanale.

**Skala robocza.** Preprocessing SAM skaluje dłuższy bok (**1280**) do 1024 i dopełnia do 1024², współczynnik **×0.8** dokładnie. Płótno robocze: 1024 × 768 (AS), 1024 × 712 (K, VAB). Oceny widoczności cienkich struktur (ściany, przegroda) wykonywane **po** tym przeskalowaniu — przy ×0.8, nie przy wycofanym ×0.948.

## IV.2. Obowiązkowa kolejność operacji

```text
 1. Wczytanie obrazu i maski instancyjnej.
 2. Deterministyczne docięcie do `load_crop_bbox` (obraz i maska synchronicznie).
 3. Konwersja do jednego kanału roboczego.
 4. Multi-scale crop (TYLKO gdy aktywny):
    a. losowanie q, b. proporcjonalne okno, c. losowa pozycja,
    d. walidacja (min. instancji), e. crop obrazu i maski,
    f. resize do wymiarów treści tego obrazu (1280×960 albo 1280×890),
    g. kontrola fragmentów, h. renumeracja ID.
 5. Transformacja orientacji D4.
 6. Kontrola geometrii i zgodności maski.
 7. Transformacje mask-aware: OneOf(pole jasności, przyciemnienie); syntetyczna przegroda.
 8. Kontrola maski, komponentów i ID.
 9. Fotometria globalna: OneOf(brightness/contrast, gamma); Gaussian blur.
10. Preprocessing modelu: resize z proporcjami (dłuższy bok → 1024), padding, normalizacja.
11. Powielenie kanału do RGB.
12. Targety pochodne: PerObjectDistanceTransform, bounding boxy, prompty.
13. Konwersja do tensorów.
```

**Reguła nadrzędna:** targety pochodne (12) generowane **po** wszystkich transformacjach zmieniających maskę.

**Konsekwencja kolejności:** crop (4) wykonuje się **przed** przegrodą (7), więc crop nigdy nie działa na wstawioną przegrodę. Przegrodę może zdegradować tylko blur (9) i stały downscaling ×0.8 (10) — oba kontrolowane kryterium akceptacji. Interakcja „przegroda + crop" jest więc bezprzedmiotowa.

---

# CZĘŚĆ V. RODZINY TRANSFORMACJI

## V.0. Przegląd

| Rodzina | Rola | Zawartość |
|---|---|---|
| F1. Orientacja | testowana rodzina (E1), zwykle w bazie | pełna grupa ośmiu symetrii D4 |
| F2. Skala | kandydat | multi-scale crop z zachowaniem proporcji |
| F3. Fotometria globalna | dwaj kandydaci | (a) OneOf(brightness/contrast, gamma); (b) Gaussian blur |
| F4. Fotometria lokalna mask-aware | jeden kandydat | OneOf(pole jasności, lokalne przyciemnienie) |
| F5. Strukturalna | kandydat | syntetyczna przegroda dzieląca por na dwie instancje |

Wszystkie parametry poniżej są **wartościami startowymi do kwalifikacji i strojenia**.

## V.1. F1 — Orientacja (grupa D4)

### Rola

Segmentacja powinna być niezależna od orientacji próbki (ekwiwariancja). Kąt anizotropii jest **mierzony z maski**, nie przewidywany, więc obrót nie zaburza pomiaru anizotropii — usuwa jedynie skrót orientacyjny. Transformacja jest bardzo tania (obroty 90° i odbicia bez interpolacji). Dla atrybucji **D4 jest testowane jako pierwsza rodzina** (E1) na tle B0; oczekiwanie: neutralne/korzystne → zostaje w bazie.

### Osiem unikalnych symetrii

```text
I, R90, R180, R270, H, H∘R90, H∘R180 (= odbicie pionowe), H∘R270
```

Nie składa się niezależnych flipów i obrotów (to tworzy duplikaty: H+V = R180).

### Implementacja

```python
A.D4(p=1.0)
```

z zapisanym seedem. Maska przechodzi identyczną geometrię (nearest-neighbor).

### Obraz prostokątny

Obrót 90° zmienia 1280×960 na 960×1280 (dla K i VAB: 1280×890 na 890×1280). **Zakaz rozciągania z powrotem** — zmieniłoby geometrię (anizotropię). Obsługuje to preprocessing modelu; dłuższym bokiem pozostaje 1280, więc współczynnik ×0.8 i udział paddingu nie zmieniają się po obrocie. Przy `batch_size = 1` różne kształty nie stanowią problemu (II.2).

### Interpretacja

Osiem orientacji = osiem widoków tej samej mikrostruktury, nie nowe dane. Porównania po **krokach optymalizatora**, nie epokach — inaczej D4 dostałby ukrytą przewagę większego wirtualnego zbioru.

## V.2. F2 — Multi-scale crop

### Intuicja i uzasadnienie

Raz wycinamy mniejszy fragment i powiększamy (pory zajmują więcej pikseli), raz większy i pomniejszamy. Model przestaje wiązać się z jednym typowym rozmiarem pora. Kontrolowane powiększanie cropów z binu `coarse` przybliża skalę binu `fine` — czyli uczy model, jak te same pianki wyglądałyby na drugim mikroskopie i przy 50×. Zachowuje proporcje i anizotropię przy izotropowym resize.

### Geometria

`q = rozmiar_docelowy / rozmiar_cropu`. Dla treści `H × W`: `H_c = round(H/q)`, `W_c = round(W/q)`, `q ≥ 1.0`. Crop zachowuje proporcje obrazu (nie kwadratowy) i jest skalowany z powrotem **do wymiarów treści tego samego obrazu** — nie do jednej stałej pary liczb, bo treść ma dwie geometrie (1280×960 dla AS, 1280×890 dla K i VAB).

Przykłady przy `q = 1.30` (maksimum zamrożone):

```text
AS      (1280 × 960):  crop 985 × 738  ->  resize do 1280 × 960
K, VAB  (1280 × 890):  crop 985 × 685  ->  resize do 1280 × 890
```

Bin `coarse` obejmuje obrazy z obu mikroskopów (K@30× i VAB1@30× są `coarse`), więc obie geometrie realnie występują w F2. **Zakaz `q < 1`.**

### Kalibracja zakresu (wykonana)

`pixel_size_um` jest rzeczywisty per obraz (III.2), więc `q_max` liczy się **per obraz**, względem najdrobniejszej skali nieodstającej w zbiorze:

```text
q_max(i) = pixel_size_um(i) / 2.480469
```

| scale_bin | pixel_size_um | q_max(i) | polityka zamrożona |
|---|---:|---:|---|
| coarse | 3.307292 | 1.333 | `q ∈ [1.00, 1.30]` |
| coarse | 3.24023 | 1.306 | `q ∈ [1.00, 1.30]` |
| fine | 2.59219 | 1.045 | `q = 1.00` |
| fine | 2.480469 | 1.000 | `q = 1.00` |
| outlier | 0.25–0.76 | 3–13 | `q = 1.00` (wyłączone) |

**Zamrożone: `q_max = 1.30` dla binu `coarse`; `q = 1.00` dla `fine` i `outlier`.** Wartość 1.045 dla AS@50× leży poniżej szumu i jest zaokrąglana do 1.00. Globalne `q_max` liczone łącznie z obrazami `scale_outlier` wynosi **13.33** — jest artefaktem sześciu zbliżeń i **nie wolno go użyć**.

**Bramka decyzyjna — wynik: ZALICZONA.** Próg: stosunek skal < ~1.10 degradowałby rodzinę F2 do opcjonalnej, ustępującej priorytetem F4/F5. Zmierzono: stosunek wewnątrz AS (40×/50×) = **1.2500**, pełny rozstęp bez outlierów = **1.3333**. Oba powyżej progu, więc **F2 zostaje w kolejce z normalnym priorytetem**, w miejscu przewidzianym w części XI.

**Zastrzeżenie do interpretacji wyniku E2.** Bin `fine` to 10.7% zbioru, a w VALIDATION będzie to rząd 10–15 obrazów. Metryka główna liczona globalnie jest zdominowana przez `coarse` i najprawdopodobniej pokaże efekt nieodróżnialny od `Δ_sig`. Dlatego decyzja o F2 opiera się przede wszystkim na **metrykach per `scale_bin`** i na **deterministycznym stress-teście skali**, a nie na globalnym instance F1. Wynik `inconclusive` na metryce głównej przy wyraźnej poprawie na binie `fine` jest podstawą do przyjęcia F2, pod warunkiem braku pogorszenia na `coarse`.

### Rozkład startowy q

```text
scale_bin = coarse:
  50%: q = 1.00
  30%: q ∈ [1.05, 1.15]
  20%: q ∈ [1.15, 1.30]
scale_bin = fine lub outlier:
  q = 1.00
```

Strojenie: ≤ 2 warianty `q_max ∈ {1.15, 1.30}` (limit z części X). Nie stroi się jednocześnie `q_max` i rozkładu.

### Pozycja okna i walidacja

**100% pozycji losowanych równomiernie.** Walidacja: crop musi zawierać **≥ 3 instancje** spełniające próg powierzchni; przy niepowodzeniu ≤ 5 ponownych losowań, potem fallback do `q = 1.0` (logowany). Przy gęstych piankach fallback powinien być rzadkością.

**Dlaczego bez dedykowanego samplera instancji.** Przy całych obrazach, `q ≤ 1.25` i gęstych piankach niemal każdy crop ma dziesiątki instancji. Wymuszanie minimum przez sampler wprowadzałoby ukryty bias gęstości; prosty próg + fallback osiąga cel bez biasu.

### Instancje przecięte krawędzią

```text
A_min_fragment = P1 (pierwszy percentyl) rozkładu powierzchni instancji GT, raz na TRAIN.
ZAMROŻONE (2026-08-25): A_min_fragment = 432.0 px^2 w pikselach zrodlowych,
z 25 253 instancji TRAIN splitu split_v1, mierzone po docieciu do
load_crop_bbox, bez obrazow scale_outlier. Diagnostyka per bin:
coarse 451.5, fine 217.8 px^2 (III.5 "Wynik wykonania").
```

Zasada: nie tworzymy etykiet mniejszych niż cokolwiek, co anotator realnie oznaczył. Fragment ≥ `A_min_fragment` i spójny — zachowany; mniejszy — usunięty; instancja dotykająca brzegu → flaga `border_instance`; gęsta renumeracja ID. **Próg stosuje się wyłącznie do instancji, którym cięcie realnie zabrało powierzchnię** — instancja nietknięta przez krawędź przeżywa w każdym rozmiarze (uzasadnienie i pomiar: IV.1 „Doprecyzowanie zakresu `A_min_fragment`"). `border_instance` wykluczone z metryk morfologicznych **spójnie po stronie GT i predykcji**.

### Interpolacja

```text
pomniejszanie: INTER_AREA,  powiększanie: INTER_LINEAR,  maska: INTER_NEAREST
```

### Logowanie (per crop)

`q`, rozmiar okna, pozycja, liczba instancji przed/po, usunięte fragmenty, instancje brzegowe, liczba prób, użycie fallbacku.

## V.3. F3a — Fotometria tonalna: OneOf(brightness/contrast, gamma)

Brightness/contrast: całe zdjęcie jaśniejsze/ciemniejsze, kontrast większy/mniejszy — odwzorowuje różnice detektora/akwizycji. Gamma: nieliniowa zmiana, mocniej modyfikuje średnie i ciemne tony — odwzorowuje różnice odpowiedzi tonalnej między mikroskopami. Geometria i maska niezmienione. Testowane łącznie jako `OneOf` (wspólny cel — odporność tonalna).

```python
A.OneOf(
    [
        A.RandomBrightnessContrast(brightness_limit=(-0.10, 0.10),
                                   contrast_limit=(-0.15, 0.15), p=1.0),
        A.RandomGamma(gamma_limit=(90, 110), p=1.0),
    ],
    p=0.5,
)
```

Prawdopodobieństwo kontenera i równe wagi zapisane jawnie. Dekompozycja (BC vs gamma osobno) tylko przy wyniku negatywnym/niejednoznacznym.

## V.4. F3b — Gaussian blur

Bardzo lekkie rozmycie — jak niewielkie niedoostrzenie/niższa efektywna rozdzielczość. Realne zdjęcia różnią się ostrością, working distance, parametrami skanowania. Był stosowany w pracy referencyjnej.

```text
kernel: 3, sigma: 0.2–0.8, p: 0.2
```

Maska nie jest rozmywana. Efekt oceniany po pełnym preprocessingu (docięcie + ×0.8). `sigma ≤ 0.8` px źródłowych to efektywnie ≤ 0.64 px roboczych — dla porów jest to bezpieczne (III.1), więc kryterium akceptacji dotyczy w praktyce **cienkich ścian**, których grubości manifest nie zna i które ocenia Faza 0. Nie może usuwać cienkich ścian ani małych porów. W indywidualnym screeningu **nie** stosowany razem z przegrodą (interakcja badana osobno w E7).

## V.5. F4 — Fotometria lokalna mask-aware: OneOf(pole jasności, przyciemnienie)

Obie celują w **fałszywy podział instancji (split)**: model nie powinien mylić powolnej zmiany jasności ani lokalnego cienia wewnątrz pora z granicą nowej instancji. Praca referencyjna opisała pogorszenie przy silnych zmianach głębi i zacienionych szczelinach.

### V.5.1. Niskoczęstotliwościowe pole jasności

Wnętrze wybranych porów łagodnie jaśniejsze/ciemniejsze albo z gradientem; granica niezmieniona. **Nie** stosować ostrego `image[mask>0] += field` — siła zero przy granicy, narasta ku rdzeniowi (distance transform).

```text
p na obraz: ≈ 0.3;  modyfikowane pory: 30–50%;  strength: 0.08–0.15 zakresu (P95−P5)
rodzaj: stałe / gradient liniowy / gładkie pole losowe (siatka 2×2 lub 3×3)
```

Tekstura zachowana; zbyt małe pory pomijane; **maska bitowo identyczna**.

### V.5.2. Lokalne przyciemnienie podregionu

Ciemniejsza plama we wnętrzu jednego pora (widok w głąb przez rozerwaną membranę); całość nadal jedną instancją.

```text
p na obraz: 0.2–0.3;  pory: 1–2;  podregion: 1;  powierzchnia: 5–20%;  factor: 0.60–0.85
```

Elipsa/wygładzony obrys, losowa orientacja, **wyłącznie w erodowanym rdzeniu**, miękka krawędź, naturalna tekstura, **bez styku z granicą**. Wariant stresowy: powierzchnia 15–30%, factor 0.45–0.70. Nie stosuje się obu mask-aware do tej samej instancji (stąd `OneOf`).

## V.6. F5 — Syntetyczna przegroda

W dużym porze generujemy wiarygodną ściankę; por staje się dwiema instancjami z dwoma ID. Celuje w **sklejanie (merge)** ciasno sąsiadujących porów. W pracy referencyjnej analogiczne dzielenie poprawiało działanie na gęstych mikrostrukturach. Separator **nie** jest prostą czarną linią — przypomina rzeczywistą ścianę SEM.

```text
p na obraz: ≈ 0.20;  przecięcia: 1 (później maks. 2);  kandydaci: górne 20–30% powierzchni
proporcja fragmentów: 0.25–0.75;  grubość: kalibrowana na realnych ścianach
w rozdzielczości roboczej (po docięciu i ×0.8), orientacyjnie 1–3 px
```

**Geometria:** linia prosta lub łagodna Bézier; początek i koniec na różnych częściach granicy; dokładnie dwa komponenty; brak odprysków.

**Procedura maski:** wybór ID → separator → usunięcie jego pikseli → connected-components tylko wybranego ID → akceptacja przy dokładnie dwóch komponentach → dwa nowe ID → gęsta renumeracja → kontrola spójności → dopiero potem targety pochodne.

**Akceptacja realizacji:** instancja dość duża; separator łączy dwie różne części granicy; dwa komponenty; oba ≥ próg; brak odprysków; dwa poprawne ID; **przegroda widoczna po pełnym preprocessingu**. Kontrolować nadsegmentację i zmianę rozkładu rozmiarów.

---

# CZĘŚĆ VI. METRYKI

## VI.1. Metryka główna

```text
micro instance F1 przy mask IoU ≥ 0.5, z dopasowaniem węgierskim
```

Mask IoU zamiast bbox IoU: pory bywają wydłużone i ukośne, ich bbox jest znacznie większy od maski, więc dopasowania po bbox byłyby zniekształcone — a anizotropia to kluczowy wynik materiałowy.

**Implementacja mask IoU** (macierz kontyngencji, jedna operacja na obraz):

```python
n_gt, n_pr = gt.max() + 1, pred.max() + 1
cont = np.bincount(gt.ravel().astype(np.int64) * n_pr + pred.ravel(),
                   minlength=n_gt * n_pr).reshape(n_gt, n_pr)
# IoU[i, j] = cont[i, j] / (area_gt[i] + area_pred[j] - cont[i, j])
```

**Dopasowanie:** `linear_sum_assignment` na `−IoU` (bez tła), odrzucenie par z IoU < 0.5. Raport: TP, FP, FN, precision, recall, F1, średni mask IoU par, względna różnica liczby porów. Zaleca się sprawdzić gotowe, przetestowane metryki w ekosystemie Micro-SAM / elf (mSA/SA50) dla porównywalności z literaturą; własna implementacja musi przejść testy VI.3.

## VI.2. Metryki zabezpieczające

1. **Boundary F1** — kontury dopasowanych instancji, tolerancje **2 px (główna)** i **5 px (diagnostyczna)** w px roboczych, z przeliczeniem na µm; kontury na krawędzi kadru wykluczone. **Uwaga interpretacyjna:** przy ×0.8 tolerancja 2 px to ~4% średnicy typowego pora, ale ~36% średnicy pora najmniejszego (III.1). Metryka jest więc mało wymagająca na małych porach — wysokiego boundary F1 w koszu najmniejszych instancji nie wolno czytać jako dowodu precyzji granic. Progi pozostają bez zmian; zmienia się wyłącznie sposób opisu wyniku.
2. **Błąd liczby porów** — podpisany `(N_pred − N_gt)/N_gt` i wartość bezwzględna.
3. **Liczniki merge i split** (z macierzy kontyngencji):

```text
merge: predykcja P sklejeniem, gdy |P∩g|/|g| ≥ 0.5 dla ≥ 2 różnych GT g
split: instancja GT g podziałem, gdy |P∩g|/|P| ≥ 0.5 dla ≥ 2 różnych predykcji P
```

To bezpośrednie metryki celów F5 (merge) i F4 (split).

4. **Recall w koszach rozmiaru instancji** — kosze = kwartyle powierzchni GT (raz na TRAIN), raportowane w µm². Właściwa metryka celu „pomijanie małych porów".
5. **Precision i recall osobno.**

Tie-break wyboru checkpointu/polityki: `F1 → boundary F1 (2 px) → |błąd liczby porów| → (merge+split) → koszt`.

## VI.3. Walidacja implementacji metryk (przed pierwszym treningiem)

Testy syntetyczne z zapisanym oczekiwaniem: predykcja idealna; jedna brakująca; jedna nadmiarowa; sklejenie dwóch porów (merge += 1); podział jednego pora (split += 1); poprawna liczba, przesunięte granice (spadek boundary F1); puste GT; pusta predykcja. Sprawdzić, że zamiana kolejności GT↔predykcja nie zamienia FP z FN ani precision z recall, a merge i split zamieniają się rolami (nie znikają).

## VI.4. Uśrednianie i raportowanie

Każda metryka: **micro** (po instancjach) i **macro** (per obraz). Przekroje: **per formulacja**, **per `scale_bin`** (`coarse` / `fine`), **per `material`** (AS / K / VAB) i **per mikroskop** (M1 / M2). Każdy wynik podzbioru z liczebnością. Obrazy `scale_outlier` są wyłączone z przekrojów i raportowane osobno. Przekrój po `scale_bin` jest obowiązkowy przy ocenie F2 (V.2), a przekrój po mikroskopie — przy ocenie F3 i F4.

## VI.5. Metryki materiałowe

Dla GT i predykcji, per obraz i zbiorczo per formulacja × `scale_bin`: liczba porów; liczba na jednostkę powierzchni; udział powierzchniowy (porowatość 2D); rozkład powierzchni; średni i medianowy rozmiar (Feret); Feret i MinFeret; anizotropia `R = Feret/MinFeret`; kąt `θ` (0–180°); odległość Wassersteina rozkładów rozmiaru. `border_instance` wykluczone spójnie po obu stronach.

### VI.5.1. Błędy anizotropii per dopasowana para

```text
błąd elongacji:  |R_pred − R_gt| / R_gt
błąd kąta:       Δθ = min(|θ_pred − θ_gt| mod 180°, 180° − |θ_pred − θ_gt| mod 180°)
```

`Δθ` tylko dla par o `R_gt ≥ 1.2` (poniżej kąt to szum). Opcjonalne ważenie przez `(R_gt − 1)`. Metryki per para nadrzędne nad średnimi populacyjnymi (te maskują kompensujące się błędy).

### VI.5.2. Agregacja kąta (średnia osiowa)

Kąt jest osiowy modulo 180°. Zwykła średnia arytmetyczna jest błędna:

```text
θ̄ = ½ · atan2( Σ sin(2θᵢ), Σ cos(2θᵢ) )
```

Dodatkowo **osiowa długość wypadkowa** `R_axial = (1/N)·√((Σcos2θᵢ)² + (Σsin2θᵢ)²)` jako miara koncentracji (niska = kąt średni niemiarodajny).

---

# CZĘŚĆ VII. FAZA 0 — WERYFIKACJA WIZUALNA AUGMENTACJI (mini-eksperyment, obowiązkowa bramka)

**To jest osobna faza wykonywana przed jakimkolwiek treningiem.** Jej celem jest ekspercka ocena, czy każda augmentacja jest racjonalna i **nie psuje rozpoznawalności obiektów** — czy po transformacji (i po pełnym preprocessingu) nadal dałoby się ten obraz wiarygodnie zaanotować. Kod służy tu do przygotowania materiału do oceny; decyzję podejmuje człowiek. Żadna transformacja nie wchodzi do właściwego eksperymentu bez przejścia tej bramki.

## VII.1. Przebieg mini-eksperymentu

```text
1. Zaimplementować transformację (pojedynczo).
2. Uruchomić skrypt generujący plansze kontrolne (VII.3) na golden gallery (VII.2),
   dla trzech poziomów siły: niski / nominalny / wysoki.
3. Ekspert przegląda plansze i nadaje status: accepted / revise / rejected (VII.4).
4. revise → korekta zakresów parametrów → powrót do kroku 2.
5. Dopiero accepted odblokowuje transformację do screeningu (część XI).
```

Inspekcję powtarza się: po implementacji; po każdej zmianie zakresu parametrów; przed testem kombinacji; okresowo podczas treningu (na zapisanych batchach); po treningu przy porównaniu predykcji.

## VII.2. Golden gallery — wyłącznie z TRAIN

Zamrożony zestaw obrazów kontrolnych **wyłącznie z formulacji treningowych** (VALIDATION dopuszczalne do porównań predykcji po treningu; TEST nigdy — wielokrotne oglądanie obrazów testowych to miękki przeciek). Pokrycie: możliwie wiele formulacji, **oba mikroskopy (M1 i M2)** — a więc i **obie geometrie treści (1280×960 i 1280×890)** — **oba biny skali (`coarse` i `fine`)**, małe i duże pory, niska i wysoka gęstość, cienkie ściany, cienie, przypadki łatwe i trudne. Plansze dla obrazów M2 muszą pokazywać obraz **po docięciu paska**, bo to jest wejście, które faktycznie widzi model. Dla F2 plansze muszą pokazywać `q ∈ {1.00, 1.15, 1.30}` na obrazach `coarse` — sprawdzane jest, czy przy `q = 1.30` cienkie ściany przeżywają resize i preprocessing.

| Typ transformacji | Liczba obrazów |
|---|---:|
| D4 | 4–6 |
| fotometria globalna | 6–8 |
| multi-scale crop | 10–12 |
| mask-aware | ≥ 12 |
| syntetyczna przegroda | 12–20 |

Dla transformacji parametrycznych: co najmniej trzy poziomy (niski/nominalny/wysoki).

## VII.3. Zawartość planszy kontrolnej (co generuje skrypt)

```text
obraz oryginalny
obraz po augmentacji
maska oryginalna
maska po augmentacji
overlay maski na obrazie po augmentacji
obraz po PEŁNYM preprocessingu modelu (docięcie + resize ×0.8 + padding + normalizacja)
wylosowane parametry augmentacji
```

Dla mask-aware dodatkowo: ID instancji, obszar modyfikacji, odległość obszaru od prawdziwej granicy.
Dla przegrody dodatkowo: separator, maska przed i po, dwa nowe ID, liczba komponentów, widoczność po pełnym preprocessingu.

Szkic działania skryptu: wczytuje golden gallery i manifest; dla wybranej transformacji i listy poziomów siły stosuje ją (z ustalonym seedem, dla powtarzalności), renderuje panele jak wyżej do plików PNG w katalogu przeglądu, dołącza JSON z wylosowanymi parametrami. Ekspert przegląda katalog i wypełnia arkusz statusów.

## VII.4. Kryteria akceptacji (rozpoznawalność jest nadrzędna)

Transformacja przechodzi, jeżeli **wszystkie** poniższe są spełnione:

- obraz nadal przypomina realny SEM z badanej domeny;
- **nadal dałoby się go wiarygodnie zaanotować** — pory pozostają rozpoznawalne;
- obraz i maska są zgodne;
- nie powstają sztuczne krawędzie niezgodne z celem transformacji;
- **cienkie ściany i małe pory nie znikają** w niekontrolowany sposób (sprawdzane po docięciu i ×0.8);
- transformacja czysto fotometryczna nie zmienia morfologii;
- maksymalna siła pozostaje wiarygodna;
- wynik po resize, paddingu i normalizacji jest poprawny.

Status: `accepted / revise / rejected`. Wynik zapisywany w arkuszu decyzji (część XVI-szablon).

## VII.5. Automatyczna kontrola integralności (uzupełnia ocenę wizualną, w locie podczas treningu)

**Wspólne:** zgodność wymiarów obrazu/maski; poprawny `dtype`; brak `NaN`/`Inf`; zakres intensywności; tło = 0; brak ujemnych/interpolowanych ID; spójność każdego dodatniego ID; poprawna renumeracja; brak niezamierzonej pustej maski.
**Fotometria:** maska **bitowo identyczna**; zmienia się tylko obraz; jeden kanał roboczy.
**Geometria (D4, crop):** obraz i maska tą samą geometrią; maska nearest-neighbor; liczba ID zmienia się tylko z uzasadnionego powodu.

**Fallback vs błąd.** Oczekiwane niepowodzenie losowania (brak przegrody; za mały por; crop bez wymaganych instancji; separator > 2 komponenty): maks. N prób → kontrolowany fallback → próbka bez transformacji → log. Prawdziwy błąd integralności (błędne ID; rozspójnienie; NaN; niezgodne wymiary; target niezgodny z maską): zatrzymuje batch/run. **Zakaz cichego pomijania próbek.**

## VII.6. Benchmark DataLoadera (tylko operacje kosztowne)

Dla crop, pola jasności, przyciemnienia, przegrody. Kolejność: testy jednostkowe → akceptacja wizualna → benchmark. 100 batchy rozgrzewki, 500–1000 pomiaru, ten sam sprzęt i workerzy, osobno baza i transformacja. Mierzone: `data_time`, `step_time`, throughput, CPU, GPU, czas oczekiwania GPU, retry/fallback rate. D4 i fotometria globalna bez osobnego benchmarku, chyba że profilowanie wskaże problem.

---

# CZĘŚĆ VIII. CHECKPOINTY, WALIDACJA I EARLY STOPPING

## VIII.1. Walidacja standardowa

Oryginalne obrazy walidacyjne; wyłącznie deterministyczny preprocessing; brak losowych augmentacji i przegród; stała inferencja AIS ze stałymi parametrami watershed. Tylko ona służy do wyboru `best_primary.ckpt`.

## VIII.2. Dwupoziomowa kadencja ewaluacji

```text
ewaluacja lekka: val loss co E kroków
ewaluacja pełna: inferencja AIS + pełny zestaw metryk co k·E kroków
```

Cel: 20–40 ewaluacji pełnych na pełny run. Wybór checkpointu i early stopping — wyłącznie na ewaluacjach pełnych. Val loss = monitoring zdrowia treningu.

## VIII.3. Checkpointy

**`last.ckpt`** (wznowienie tego samego runu): model, LoRA, dekoder AIS, optymalizator, scheduler, gradient scaler, krok/epoka, RNG, stan samplera.
**`best_primary.ckpt`**: najlepsza ewaluacja pełna wg metryki głównej (VI.1) na walidacji standardowej; tie-break jak VI.2.

## VIII.4. Early stopping

Aktywny **wyłącznie w runach 100% `T_full`** (w tym po promocji). W screeningu runy biegną do budżetu i porównywane są przy **równej liczbie kroków** — aktywny ES rozjeżdżałby momenty zatrzymania. Patience w liczbie ewaluacji pełnych.

---

# CZĘŚĆ IX. BUDŻET, BAZA I PRÓG ISTOTNOŚCI

## IX.1. Jednostka porównania

**Krok optymalizatora**, nie epoka. Epoki raportowane pomocniczo.

## IX.2. Baza kalibracyjna B0 = brak augmentacji

```text
B0: brak jakiejkolwiek augmentacji (tylko deterministyczny preprocessing),
    trzy pełne seedy, trening z bezpiecznie wysokim limitem kroków.
```

B0 pełni dwie role: (1) referencja tabeli atrybucji („augmentacja X vs jej brak"), (2) kalibracja. Wyznacza: wariancję między seedami; moment plateau; **`T_full`** (obejmujący zbieżność + zapas na patience; jeśli bogate polityki zbiegają później, `T_full` dobrać z zapasem obejmującym najbogatszą politykę — zweryfikować na D4 i FULL); **próg istotności**:

```text
Δ_sig = rozstęp (max − min) metryki głównej z trzech seedów B0
        (jeśli rozstęp < ~0.002: Δ_sig = max(rozstęp, 2·σ̂))
```

## IX.3. Dwa budżety

```text
SCREEN = 60% T_full     (kandydaci przesiewowi)
FULL   = 100% T_full    (promowani, FULL, FINAL)
```

Każdy run od początku ma `max_steps = T_full` i `scheduler_horizon = T_full`. Screening = zatrzymanie w 60%. Promocja = **wznowienie tego samego runu** (pełny stan) do 100%. Wczesne ubijanie przed 60% tylko przy katastrofie (błąd integralności; pierwsza ewaluacja pełna drastycznie poniżej bazy).

---

# CZĘŚĆ X. REGUŁY DECYZJI I LIMITY RUNÓW

## X.1. Sparowane porównania

Kandydat vs aktualna baza przy identycznych: seedzie, kolejności obrazów, splicie, liczbie kroków, harmonogramie ewaluacji. Losowość augmentacji odtwarzalna przez seed.

## X.2. Statusy

**Pozytywny** (brak błędów integralności, akceptowalny koszt): `Δ ≥ +Δ_sig`; albo `Δ ≥ −Δ_sig` przy jednoznacznej poprawie docelowego błędu (merge/split/recall małych/boundary F1) przekraczającej jego rozstęp z B0, na więcej niż jednym obrazie.
**Negatywny:** `Δ < −Δ_sig` bez poprawy celu; albo niewiarygodność wizualna; albo błędy masek; albo wysoki koszt bez uzysku.
**Niejednoznaczny:** wynik w paśmie `±Δ_sig`; albo poprawa jednego aspektu kosztem innego. Otrzymuje: promocję do 100%; drugi sparowany seed jeśli nadal blisko granicy; najwyżej jeden alternatywny wariant parametrów.

## X.3. Dekompozycja OneOf

F3-tonalna i F4-mask-aware testowane od razu jako `OneOf`. Dekompozycja (≤ 2 runy) tylko przy statusie negatywnym/niejednoznacznym.

## X.4. Strojenie

Po pozytywnym screeningu ≤ 2 dodatkowe warianty (słabszy/mocniejszy). Druga oś strojenia tylko przy wyraźnej czułości pierwszej.

## X.5. Beam width 2

Dwie polityki różniące się < `Δ_sig` można zachować jako równoległe bazy (maks. dwie).

## X.6. Limity seedów

| Etap | Seedy |
|---|---:|
| Baza B0 | 3 |
| Screening kandydata (w tym D4) | 1 |
| Kandydat blisko granicy | +1 |
| Zwycięzca etapu pośredniego | ≤ 2 |
| Ablacja leave-one-family-out | 1 (więcej dla rodzin borderline) |
| B0 / FULL / FINAL na TEST | 3 |
| Grupowy k-fold (opcjonalny) | 1–2 na fold |

Trzy seedy rezerwuje się dla B0 i porównań na TEST, nie po każdej rodzinie.

---

# CZĘŚĆ XI. SEKWENCJA EKSPERYMENTÓW

Baza narasta rodzinami. `BASE_X` = zamrożony zwycięzca do danego etapu. Decyzje na VALIDATION; TEST nietknięty do końca.

```text
─────────────────────────────────────────────────────────────────────────
FAZA 0: WERYFIKACJA WIZUALNA (część VII) — obowiązkowa bramka.
   Każda transformacja: implementacja → plansze → ocena ekspercka →
   accepted/revise/rejected. Tylko accepted wchodzi do screeningu.
─────────────────────────────────────────────────────────────────────────
P0 (opcjonalny): wybór bazowego checkpointu.
   2 krótkie runy (natywny SAM vs generalista EM), ~30% T_full, jeden seed.
─────────────────────────────────────────────────────────────────────────
E0: BAZA = brak augmentacji (B0). Trzy pełne seedy.
   Wyznacza: wariancję, T_full, Δ_sig, plateau, kadencję (E, k), patience.
   → BASE_NONE = B0
─────────────────────────────────────────────────────────────────────────
E1: ORIENTACJA (D4). O1 = BASE_NONE + D4. Test dla atrybucji.
   Porównanie po krokach optymalizatora. Oczekiwane: neutralne/korzystne.
   → BASE_ORIENTATION = D4 (albo B0, gdyby D4 zaszkodziło)
─────────────────────────────────────────────────────────────────────────
E2: SKALA. Bramka ZALICZONA (stosunek 1.25–1.33, V.2) — F2 zostaje w kolejce.
   S1 = BASE_ORIENTATION + multi-scale crop, q_max = 1.30 na binie coarse.
   Odczyty: per scale_bin (obowiązkowo) i per mikroskop; stress-test skali.
   Uwaga: bin fine to ~10–15 obrazów w VAL — metryka globalna może być
   nierozstrzygająca; decyduje przekrój per scale_bin (V.2).
   Strojenie: ≤ 2 warianty q_max ∈ {1.15, 1.30}.  → BASE_SCALE
─────────────────────────────────────────────────────────────────────────
E3: FOTOMETRIA TONALNA. P_ton = BASE_SCALE + OneOf(BC, gamma).
   Kolejność E2 przed E3 ZAMROŻONA (uzasadnienie pod diagramem).
   Dekompozycja warunkowa.  → BASE_TON
─────────────────────────────────────────────────────────────────────────
E4: BLUR. P_blur = BASE_TON + Gaussian blur.  → BASE_GLOBAL
─────────────────────────────────────────────────────────────────────────
E5: FOTOMETRIA LOKALNA. M = BASE_GLOBAL + OneOf(pole, przyciemnienie).
   Odczyty: split, liczba instancji, boundary F1, cienie, kosze małych porów.
   Dekompozycja warunkowa.  → BASE_MASK
─────────────────────────────────────────────────────────────────────────
E6: PRZEGRODA. T1 = BASE_MASK + syntetyczna przegroda.
   Odczyty: merge, recall małych, rozkład rozmiarów. Screening BEZ blur.
   Logować faktyczny odsetek próbek z przegrodą (podstawa porównania).
   → BASE_STRUCT
─────────────────────────────────────────────────────────────────────────
E7: INTERAKCJA (obowiązkowa): przegroda + blur.
   Czy separator czytelny po blur + ×0.8. Decyzja po inspekcji i treningu.
─────────────────────────────────────────────────────────────────────────
E8: FULL = D4 + [skala] + [tonalna] + [blur] + [mask-aware] + [przegroda]
   (nawiasy = rodziny z wynikiem pozytywnym). Pełny run + stress-testy (XI.1).
─────────────────────────────────────────────────────────────────────────
E9: PEŁNA ABLACJA LEAVE-ONE-FAMILY-OUT.
   Dla KAŻDEJ rodziny w FULL: wariant FULL bez tej rodziny.
   1 seed, 60% T_full; promocja do 100% dla rodzin bliskich Δ_sig.
   Drugi widok atrybucji ("wkład przy usunięciu").
─────────────────────────────────────────────────────────────────────────
FINAL: zamrożenie polityki → ocena na TEST → model wdrożeniowy (część XII).
─────────────────────────────────────────────────────────────────────────
```

**Kolejność E2 przed E3 — decyzja zamrożona.** Rozważano zamianę, bo obecność dwóch mikroskopów jest mocnym uzasadnieniem dla rodziny fotometrycznej F3. Kolejność **zostaje bez zmian**, z trzech powodów: (1) skala jest właściwością geometryczną wejścia i naturalnie poprzedza fotometrię w kolejności pipeline'u (IV.2), więc sekwencja atrybucji jest zgodna z kolejnością przetwarzania; (2) F2 ma zamrożoną, zmierzoną kalibrację (bramka zaliczona, V.2), więc jest kandydatem o najlepiej udokumentowanym uzasadnieniu i powinien wejść do bazy wcześniej; (3) atrybucja sekwencyjna jest z definicji zależna od kolejności, a drugi widok — ablacja leave-one-family-out (E9) — mierzy wkład niezależnie od niej, więc ewentualna asymetria jest w raporcie widoczna. Zamiana wymagałaby osobnego uzasadnienia w publikacji i nie daje przewagi metodologicznej.

**Dwa widoki atrybucji.** Sekwencja E1–E6 mierzy przyrost rodziny **przy dodaniu**; ablacja E9 — stratę **przy usunięciu** z FULL. Duży dodatek + mała strata = redundancja; mały dodatek + duża strata = synergia. Oba raportowane względem B0.

## XI.1. Deterministyczne stress-testy (tylko FULL i finały)

Kopie walidacji z ustalonymi zaburzeniami: osiem orientacji D4; ustalone jasność/kontrast; gamma; blur; skala. Identyczne dla każdego modelu. Diagnoza odporności, nie wybór checkpointu. Tylko dla FULL i polityk finalnych.

---

# CZĘŚĆ XII. FINALIZACJA I OCENA KOŃCOWA

## XII.1. Zamrożenie polityki

Po E0–E9 zamraża się: listę transformacji, kolejność, prawdopodobieństwa, zakresy, sampler, preprocessing, metryki, postprocessing, split (listy formulacji per zbiór), inferencję. Powstaje `FINAL_A`, opcjonalnie `FINAL_B` (jeśli różnica < `Δ_sig`).

## XII.2. Ocena na zbiorze TEST

TEST otwierany **raz**, po zamrożeniu. Na TEST ocenia się `FINAL_A` (i ewentualnie `FINAL_B`) oraz — dla tabeli atrybucji — `B0` i `FULL`, każdy z **3 seedów** (trening na TRAIN+VALIDATION, ocena na TEST). Raport: wynik per seed, średnia, zmienność, wszystkie przekroje (formulacja, `scale_bin`, rodzina, mikroskop), metryki materiałowe. **Zakaz** wyboru seedu na podstawie TEST — raportuje się cały rozkład.

Ponieważ TEST był wydzielony grupowo od początku i nietknięty, liczby są nieobciążone — nie ma potrzeby korekt ani przypisów o skażeniu.

## XII.3. Opcjonalna robustność (grupowy k-fold)

Dla `FINAL_A` (i konkurenta) można dodatkowo wykonać grupowy k-fold na TRAIN+VALIDATION (III.6), by podać rozkład wyniku decyzyjnego zamiast jednego punktu VALIDATION. TEST pozostaje osobny. Wskazane dla publikacji przy dolnym końcu X.

## XII.4. Model wdrożeniowy

Po ocenie na TEST trenuje się finalny model na **wszystkich 31 formulacjach** (z grupowo wydzieloną walidacją do early stoppingu, albo bez ES przy `T_full` z kalibracji). Oszacowanie z TEST pozostaje uczciwą deklaracją jakości na niewidzianych formulacjach.

**F-ViT-H (opcjonalny):** przetrening `FINAL_A` na ViT-H (1–3 seedy) — większa pojemność kosztem wolniejszej inferencji, przy zamrożonej polityce.

## XII.5. Raportowanie atrybucji (dla publikacji)

**Tabela główna:** wiersze = B0 (referencja), kolejne bazy z sekwencji (D4, +skala, +tonalna, +blur, +mask-aware, +przegroda = FULL) oraz warianty ablacyjne (FULL − rodzina). Kolumny = metryka główna + metryki celowane (merge dla przegrody; split i recall małych dla mask-aware; boundary F1; metryki materiałowe). Każda wartość z rozkładem po seedach. Podstawa liczbowa: TEST (oraz opcjonalnie k-fold).

**Dwa widoki wkładu** (przy dodaniu vs przy usunięciu), interpretacja rozbieżności jako redundancji/synergii. **Istotność** względem `Δ_sig` i rozkładu po seedach. **Sub-składowe** (BC vs gamma; pole vs przyciemnienie) tylko jeśli wymagane — osobne ablacje z FULL; domyślnie atrybucja na poziomie rodzin.

---

# CZĘŚĆ XIII. LOGOWANIE I ODTWARZALNOŚĆ

## XIII.1. Konfiguracja runu

```yaml
run_id: ...
parent_policy: ...
candidate_transform: ...
parameter_config: ...
seed: ...
split_id: screening | test | fold k
model_backbone: ViT-L | ViT-H
base_checkpoint_hash: ...
lora_config: ...
optimizer_config: ...
scheduler_config: ...
scheduler_horizon: = T_full
input_preprocessing: ...
sampler_config:                     # sampler_run_metadata(), III.7
  split_id: split_v1
  strategy: proportional_no_oversampling
  ordering: epoch_permutation
  n_images: 494                     # = steps_per_epoch
  run_seed: ...                     # zasiew permutacji, osobny od augmentacji
  exposure: {material: ..., scale_bin: ...}
inference_mode: ais
watershed_params: ...
T_full: ...
current_budget: 60% | 100%
eval_cadence: E, k
git_commit: ...
library_versions: ...
hardware: ...
```

## XIII.2. Przebieg i wyniki (per ewaluacja)

Train/val loss; metryka główna; precision i recall osobno; boundary F1 (2/5 px); podpisany i bezwzględny błąd liczby porów; merge i split; recall per kosz; wyniki per formulacja / `scale_bin` / rodzina / mikroskop; metryki materiałowe; `data_time`; `step_time`; obrazy/s; faktyczne parametry augmentacji; faktyczna częstość transformacji niestandardowych; galerie batchy; predykcje na stałych przykładach (TRAIN + VALIDATION). Dla próbek o największym lossie — zachowane parametry augmentacji.

## XIII.3. Wizualizacja

**W trakcie treningu:** obraz przed/po augmentacji, po preprocessingu, maska, targety pochodne, nazwa i parametry.
**Po treningu** (baza vs kandydat, TRAIN + VALIDATION): obraz, GT, obie predykcje, overlay błędów z wyróżnieniem merge/split. Grupy: oba biny skali; oba mikroskopy; każda formulacja; największe poprawy; największe pogorszenia; przypadki reprezentatywne dla metryk materiałowych.

---

# CZĘŚĆ XIV. KRYTERIA ZAKOŃCZENIA

1. wykonano Fazę 0 — każda transformacja ma status accepted/revise/rejected;
2. wytrenowano B0 (3 seedy) jako referencję atrybucji;
3. testy syntetyczne metryk (VI.3) zaliczone przed pierwszym treningiem;
4. każdy kandydat (w tym D4) przeszedł screening; pozytywni — pełny budżet;
5. interakcja przegroda + blur rozstrzygnięta;
6. wykonano pełną ablację leave-one-family-out z FULL;
7. B0, FULL i FINAL ocenione na nietkniętym TEST (3 seedy);
8. sporządzono tabelę atrybucji (dwa widoki, względem B0);
9. wyniki obejmują segmentację, granice, merge/split, liczbę porów i metryki materiałowe;
10. wytrenowano model wdrożeniowy na wszystkich formulacjach;
11. każdy run odtwarzalny z konfiguracji i checkpointu.

---

# CZĘŚĆ XV. MACIERZ RUNÓW (RDZEŃ)

| Etap | Runy | Budżet |
|---|---|---|
| Faza 0 (weryfikacja wizualna) | 0 treningowych (tylko generowanie plansz) | — |
| P0 pilotaż checkpointu (opcjonalny) | 2 | ~30% |
| E0 baza B0 | 3 seedy | pełny |
| E1 orientacja D4 | 1 | 60% → 100% |
| E2 skala | 1 (+ ≤ 2 strojenie) | 60% → 100% |
| E3 fotometria tonalna (OneOf) | 1 (+ ≤ 2 dekompozycja) | 60% → 100% |
| E4 blur | 1 | 60% → 100% |
| E5 mask-aware (OneOf) | 1 (+ ≤ 2 dekompozycja) | 60% → 100% |
| E6 przegroda | 1 | 60% → 100% |
| E7 interakcja przegroda + blur | 1 | 60% → 100% |
| E8 FULL | 1 | pełny |
| E9 ablacja leave-one-family-out | tyle, ile rodzin w FULL (≈ 4–6) | 60% (→ 100% warunkowo) |
| Ocena na TEST: B0, FULL, FINAL_A | 3 polityki × 3 seedy | pełny |
| FINAL_B (warunkowy) | 3 seedy | pełny |
| Grupowy k-fold (opcjonalny) | k × (1–2) | pełny |
| Model wdrożeniowy | 1 | pełny |
| F-ViT-H (opcjonalny) | 1–3 | pełny |

Promocja pozytywnego kandydata = wznowienie tego samego runu z 60% do 100%, nie nowy run. Rdzeń bez opcji: ~18–24 runów screeningowo-budujących i ablacyjnych + 9 runów oceny na TEST (3 polityki × 3 seedy). Prawdziwy, wydzielony TEST czyni ocenę końcową tanią i nieobciążoną — bez rozmnażania foldów.

---

# CZĘŚĆ XVI. TABELA DECYZJI

## XVI.1. Decyzje podjęte (zamrożone)

| Element | Decyzja |
|---|---|
| Cel | podwójny: najlepszy model + publikowalna atrybucja |
| Tryb inferencji | AIS (`with_segmentation_decoder=True`) |
| Backbone | ViT-L (opcjonalny finał ViT-H) |
| Batch size | 1 |
| Gradient accumulation | 1 (opcjonalnie 2–4, zamrożone od startu) |
| `n_objects_per_batch` | 25 |
| Rozdzielczość źródłowa / robocza | treść 1280×960 (AS) i 1280×890 (K, VAB) / dłuższy bok 1280 → 1024 + padding do 1024², **×0.8** |
| Docięcie paska informacyjnego | zamrożone `load_crop_bbox` per mikroskop: M1 → (0,0,1280,960), M2 → (0,0,1280,890); w dataloaderze, synchronicznie obraz + maska; także w preprocessingu inferencji |
| Detektor paska | nie uczestniczy w treningu; `content_bbox` służy do jednorazowej asercji przy budowie manifestu |
| Anotacja wykraczająca poza ramkę docięcia | cięta prostą wzdłuż krawędzi; fragment zachowany przy spójności i powierzchni ≥ `A_min_fragment`, flaga `border_instance` |
| Trening na całych obrazach (bez patchy) | **potwierdzony po korekcie skali**: najmniejszy por 5.49 px roboczo, najmniejsza mediana per obraz 46.93 px roboczo (III.1) |
| Kanały | 1 roboczy → RGB na końcu |
| Metryka główna | instance F1 @ mask IoU ≥ 0.5, dopasowanie węgierskie |
| Tolerancje boundary F1 | 2 px (główna), 5 px (diagnostyczna), w px roboczych |
| Metadane | automatyczny manifest: nazwy plików + `pixel_size_um` per obraz; kolumny pochodne `material`, `microscope`, `microscope_source`, `scale_bin`, `scale_outlier`, `q_max_i`, `load_crop_bbox` |
| Nazwa kolumny rodziny | **`material`** (nie tworzy się kolumny `family`) |
| Źródło `microscope` | sidecar SEM (TM3000 → M1, SU8000 → M2); iloczyn `pixel_size_um × magnification` wyłącznie jako test spójności |
| Reguła `scale_bin` | bezwzględna, progi 3.0 i 2.4; reguła względna zdegradowana do diagnostyki |
| Tagi diagnostyczne | brak (przekroje: per formulacja / `scale_bin` / `material` / mikroskop) |
| Jednostka grupowania splitu | formulacja; `AS1`/`AS1A` i `VAB1`/`VAB11` = formulacje odrębne |
| Luki w numeracji formulacji | luki nazewnictwa, nie zgubione dane; populacja = eksport z Label Studio |
| Podział | grupowy TRAIN/VAL/TEST ze stratyfikacją po `material × scale_bin`; ≥1 formulacja M2 w VAL i TEST; oba biny skali w każdym zbiorze |
| Ocena końcowa | nietknięty TEST (3 seedy); opcjonalny grupowy k-fold |
| Baseline atrybucji | B0 = brak augmentacji, 3 seedy |
| Ablacja | pełna leave-one-family-out z FULL |
| Model wdrożeniowy | trening na wszystkich 31 formulacjach |
| Sampler obrazów | proporcjonalny, bez oversamplingu, zamrożony; permutacja epoki, własny strumień RNG z `(run_seed, epoch)` — **zaimplementowane**, III.7 |
| Odczyt zbioru TEST | wyłącznie przez `load_split(..., allow_test=True)`, z wpisem WARNING w logu — **zaimplementowane**, III.7 |
| Min. instancji w cropie | 3; do 5 prób; fallback q = 1.0 |
| `A_min_fragment` | P1 powierzchni instancji GT z TRAIN |
| Zakres q skali | `q_max = 1.30` dla `coarse`, `q = 1.00` dla `fine` i `outlier` (bramka zaliczona: 1.25–1.33) |
| Zmienna skali | `pixel_size_um` per obraz; **powiększenie wycofane** ze stratyfikacji i przekrojów |
| Rodzina pianki | identyfikowalna z prefiksu formulacji (AS / K / VAB); skonfundowana z mikroskopem |
| Obrazy `scale_outlier` | 6 sztuk; TRAIN z `q = 1.00`, wyłączone z VAL/TEST, kalibracji i przekrojów |
| Próg istotności Δ_sig | rozstęp metryki głównej z 3 seedów B0 |
| Budżety | 60% / 100% T_full; `scheduler_horizon = T_full` |
| Early stopping | tylko runy 100%; patience w ewaluacjach pełnych |
| D4 | testowane jako pierwsza rodzina (E1), zwykle w bazie |
| Fotometria tonalna / mask-aware | OneOf, dekompozycja warunkowa |
| Interakcja przegroda + crop | bezprzedmiotowa (kolejność pipeline) |
| Weryfikacja wizualna | Faza 0 — obowiązkowa bramka przed treningiem |
| Kolejność E2 i E3 | zamrożona: skala przed fotometrią tonalną (uzasadnienie w XI) |

## XVI.2. Wielkości wyznaczane proceduralnie (brak decyzji otwartych)

**Ta sekcja nie zawiera pytań do rozstrzygnięcia.** Wymienione niżej wielkości nie są wyborem metodologicznym — są odczytem albo wynikiem zamrożonej procedury, którą wykonuje się w określonym momencie kolejności wykonawczej (część XVII). Sposób ich ustalenia jest zamrożony; zmienia się wyłącznie moment, w którym pojawia się konkretna liczba.

| Wielkość | Zamrożona procedura wyznaczenia | Kiedy |
|---|---|---|
| Commit PEFT-SAM / Micro-SAM | odczyt SHA aktualnego commitu w chwili zamrożenia stosu; zapis do metadanych każdego runu | etap G |
| Bazowy checkpoint | **natywny SAM**; opcjonalny pilotaż P0 może go zastąpić generalistą EM, o ile wygra przy jednym seedzie na ~30% `T_full` (II.4) | etap F |
| Moduły LoRA / rank / alpha / dropout | **domyślne peft-sam**, przyjęte bez modyfikacji i zamrożone; odczyt wartości z konfiguracji biblioteki | etap G |
| Optymalizator / scheduler / LR | **domyślne peft-sam dla batch 1**, przyjęte bez przestrajania | etap G |
| Funkcja kosztu i wagi | **domyślna Micro-SAM dla AIS** (Dice + BCE + regresja transformat), wagi domyślne | etap G |
| Parametry watershed | kalibracja na TRAIN wg zamrożonej procedury, następnie zamrożenie na wszystkie runy | etap G |
| `T_full`, `patience`, `E`, `k` | wyznaczane z trzech seedów B0: `T_full` = zbieżność + zapas na patience, zweryfikowany na D4 i FULL; kadencja dobrana pod 20–40 ewaluacji pełnych (VIII.2) | etap L |
| `Δ_sig` | rozstęp metryki głównej z trzech seedów B0 (IX.2) | etap L |
| `A_min_fragment` | P1 rozkładu powierzchni instancji GT, liczony raz na TRAIN (V.2). **Wyznaczone: 432.0 px²** z 25 253 instancji, po docięciu do `load_crop_bbox`, bez obrazów `scale_outlier` | wykonane |
| Podział TRAIN/VAL/TEST | **21 / 5 / 5 formulacji**, grupowo, ze stratyfikacją po `material × scale_bin`, z zapisanym seedem podziału; twarde warunki z III.4 weryfikowane po losowaniu. **Wykonane: `split_v1`, seed 20260825** (III.5 "Wynik wykonania") | wykonane |
| Wysokość paska dla M2 | **ustalona: 70 wierszy**, `load_crop_bbox = (0, 0, 1280, 890)` (III.1, IV.1) | wykonane |
| Zasięg adnotacji K/VAB w obszar panelu | zmierzony na manifeście v2: **108/108 obrazów M2 (100%), 404 instancje** (~3.7/obraz tam, gdzie występuje) — cięcie poligonów z IV.1 jest obowiązkowe dla całej podserii, nie opcjonalne (IV.1 "Weryfikacja empiryczna") | wykonane |

Pozycje rozstrzygnięte wcześniej, zachowane dla śladu decyzyjnego:

| Pytanie | Rozstrzygnięcie |
|---|---|
| Wartości `pixel_size` 40× / 50× | 8 wartości per obraz, dwa klastry, stosunek 1.25–1.33 (III.1) |
| Rozkład 50× między formulacjami | 20 formulacji, wszystkie AS, po 2–3 obrazy, razem 43 (III.1) |
| Czy rodzaj pianki jest identyfikowalny | tak: AS / K / VAB z prefiksu, kolumna `material` (III.1) |
| `AS1` vs `AS1A`, `VAB1` vs `VAB11` | **odrębne formulacje**, odrębne jednostki splitu (III.3) |
| Luki w numeracji (`K4`, `AS2`, `AS8`–`AS14`) | luki nazewnictwa; populacją jest eksport z Label Studio (III.1) |
| Kolejność E2 i E3 | bez zmian: skala przed fotometrią tonalną (XI) |
| Nazwa kolumny rodziny | `material`, bez tworzenia `family` (nagłówek, III.2) |
| Rozdzielczość źródłowa i współczynnik preprocessingu | 1280 px dłuższy bok, **×0.8**; pierwotne 960×1080 i ×0.948 wycofane (III.1) |
| Czy trening na całych obrazach pozostaje zasadny po korekcie skali | tak, potwierdzone pomiarem (III.1) |

## XVI.3. Szablon decyzji dla transformacji (Faza 0 + screening)

```yaml
candidate: gaussian_blur
family: F3b_global_blur
base_policy: ...
faza0_wizualna:
  status: accepted | revise | rejected
  poziomy_sily: [niski, nominalny, wysoki]
  rozpoznawalnosc_zachowana: true | false
  approved_parameters: {kernel: 3, sigma: [0.2, 0.8], p: 0.2}
integrity_tests:
  status: passed
screening:
  seed: ...
  budget: 60% | 100%
  steps: ...
  primary_delta: ...
  boundary_delta: ...
  pores_diff_delta: ...
  merge_delta: ...
  split_delta: ...
  results_per_formulation: ...
  results_per_scale_bin: ...
  results_per_material: ...
  results_per_microscope: ...
  data_time_overhead: ...
decision: accepted | rejected | inconclusive
reason: ...
```

---

# CZĘŚĆ XVII. KOLEJNOŚĆ WYKONAWCZA

```text
A. [WYKONANE] Inwentaryzacja: pixel_size per obraz, rodziny, powiększenia.
B. [WYKONANE] Kalibracja q (q_max = 1.30) i bramka E2 (zaliczona);
   rozkład 50× ustalony (20 formulacji AS, 43 obrazy).
C. [WYKONANE] Rodzina pianki identyfikowalna (AS/K/VAB); dwa mikroskopy potwierdzone.
C2. [WYKONANE] AS1/AS1A i VAB1/VAB11 = formulacje odrębne; luki w numeracji
    to luki nazewnictwa (III.1, III.3).
C3. [WYKONANE] Geometria zweryfikowana: treść 1280×960 (AS) i 1280×890 (K, VAB);
    współczynnik preprocessingu ×0.8; pasek M2 = 70 wierszy (III.1).
C4. [WYKONANE] Rozmiar porów w px roboczych zmierzony; trening na całych
    obrazach potwierdzony (III.1).
C1. [WYKONANE] Domknąć manifest (jedna zmiana wersji, wszystkie kolumny naraz):
    material (bez tworzenia family), microscope + microscope_source,
    scale_bin, scale_outlier wg reguły bezwzględnej, q_max_i,
    load_crop_bbox; asercja content_bbox == load_crop_bbox; zapis progów,
    stałej referencyjnej q i map do metadanych runu (III.2).
C5. [WYKONANE] Sprawdzono maksymalną współrzędną y wierzchołków poligonów
    w seriach K i VAB (manifest v2, kolumna n_instances_below_crop_bbox).
    Wynik: 108/108 obrazów M2 (100%) ma co najmniej jedną instancję
    sięgającą w obszar panelu, 404 instancje łącznie (~3.7/obraz tam,
    gdzie występuje) — patrz IV.1 "Weryfikacja empiryczna". Reguła cięcia
    prostą z IV.1 jest więc obowiązkowa dla całej podserii K/VAB, nie
    opcjonalna; do zaimplementowania w dataloaderze razem z docięciem
    paska (krok G).
D. [WYKONANE] Grupowy podział TRAIN/VAL/TEST — split `split_v1`,
   seed 20260825, 21/5/5 formulacji, 494/107/106 obrazów
   (69.6% / 15.3% / 15.1% obrazów oceny). Wszystkie twarde warunki
   z III.4 spełnione; zero obrazów `scale_outlier` straconych.
   A_min_fragment = 432.0 px². Szczegóły i pełny raport: III.5
   "Wynik wykonania".
   Bramka na TEST: domknięta w kroku E (`load_split`, III.7) — strażnikiem
   jest czytnik pliku splitu, a nie dataloader, więc nie trzeba było
   czekać na krok G.
   UWAGA: dotychczasowy skrypt dzielący per obraz nie może być użyty —
   dzieli bez grupowania po formulacji, czyli generuje przeciek (III.3).
   Wyniki wcześniejszych eksperymentów uzyskane na tamtym podziale nie są
   porównywalne z tym eksperymentem. Skrypt
   `data_prep/split_dataset_into_subsets.py` jest oznaczony jako
   DEPRECATED; obowiązujący jest `scripts/create_dataset_split.py`.
E. [WYKONANE] Zamrożenie samplera obrazów: proporcjonalny, bez
   oversamplingu, permutacja epoki, własny strumień RNG zasiewany
   z `(run_seed, epoch)`. Ekspozycja na `split_v1`: AS 85.4%,
   K 9.1%, VAB 5.5%, `fine` 10.3%; epoka = 494 kroki.
   Przy okazji domknięta bramka na TEST (`load_split`). Szczegóły
   i oba nowe rozstrzygnięcia: III.7 "Wynik wykonania".
F. [DECYZJA 2026-08-25] P0 **zostaje w planie, ale wykonywany po
   zbudowaniu stosu treningowego**, jako pierwszy test dymny
   pipeline'u. Uzasadnienie i rozwiązanie cyklu zależności: II.4
   "Uwaga o kolejności".
G. Zamrożenie Micro-SAM + LoRA + dekodera AIS + preprocessingu (w tym docięcia
   do load_crop_bbox i skali ×0.8) + postprocessingu.
H. Implementacja i testy syntetyczne metryk (mask IoU, merge/split, boundary, anizotropia).
I. Implementacja transformacji + testy jednostkowe integralności.
J. FAZA 0: golden gallery (TRAIN) + skrypt plansz + ekspercka ocena wizualna (bramka).
K. Benchmark kosztownych transformacji.
L. E0: baza B0 × 3 seedy → T_full, Δ_sig, patience, kadencja.
M. E1: orientacja D4 (test na tle B0).
N. E2–E6: screening 60% → promocje 100%, budowanie bazy rodzinami.
O. E7: interakcja przegroda + blur.
P. E8: FULL + stress-testy.
Q. E9: pełna ablacja leave-one-family-out.
R. Zamrożenie polityki FINAL (+ FINAL_B warunkowo).
S. Ocena na TEST: B0, FULL, FINAL_A (+ FINAL_B); opcjonalny grupowy k-fold.
T. Tabela atrybucji (dwa widoki, względem B0).
U. Model wdrożeniowy na wszystkich formulacjach.
V. (Opcjonalnie) F-ViT-H, diagnostyka per rodzina i per mikroskop.
W. Raport: segmentacja, granice, merge/split, liczba porów, metryki materiałowe,
   atrybucja, analiza wizualna, wyniki per formulacja, scale_bin, rodzina i mikroskop.
```

---

# CZĘŚĆ XVIII. MATERIAŁY ODNIESIENIA

1. Y.-C. Cheng, Z.-Y. Tsai, C.-Y. Bair, S.-K. Yeh, C.-S. Chen, *Automatic pore characterization in SEM images of foams using a fine-tuned segment anything model*, Materials & Design 262 (2026) 115529.
2. Repozytorium `computational-cell-analytics/peft-sam` oraz dokumentacja Micro-SAM (tryb AIS, `PerObjectDistanceTransform`, `n_objects_per_batch`, `with_segmentation_decoder`).
3. E.J. Hu i in., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 (2021).
4. A. Kirillov i in., *Segment Anything*, ICCV 2023.
5. Albumentations: dokumentacja `D4`, `RandomBrightnessContrast`, `RandomGamma`, `GaussianBlur` oraz przewodniki *Choosing Augmentations*, *Semantic Segmentation*, *Targets*.