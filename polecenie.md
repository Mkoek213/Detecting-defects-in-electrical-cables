## Zadanie dla agenta

Przygotuj kompletne, przykładowe rozwiazanie do zadania segmentacji wad kabli elektrycznych na obrazach ze zbioru `data/cable`.

### Cel

Dla obrazu RGB o rozmiarze `1024 x 1024` nalezy przewidziec binarna maske wady:

- `0` = brak wady
- `255` = wada

Model ma byc zgodny z interfejsem:

```python
def predict(image: np.ndarray) -> np.ndarray:
    ...
```

Wejscie:

- `image.shape == (H, W, 3)`
- `dtype == uint8`
- przestrzen barw: RGB

Wyjscie:

- maska `shape == (H, W)`
- `dtype == uint8`
- wartosci tylko `0` albo `255`

### Klasy wad

Nalezy obsluzyc nastepujace typy uszkodzen:

- `bent_wire`
- `cable_swap`
- `combined`
- `cut_inner_insulation`
- `cut_outer_insulation`
- `missing_cable`
- `missing_wire`
- `poke_insulation`

### Bardzo wazna uwaga o danych

Foldery `train` i `test` w `data/cable` **nie sa gotowym podzialem do trenowania i walidacji projektu**:

- `data/cable/train/good` zawiera tylko obrazy poprawnych kabli
- `data/cable/test/<klasa>` zawiera obrazy uszkodzone z podzialem na klasy
- `data/cable/test/good` zawiera dodatkowe obrazy poprawne
- maski dla obrazow uszkodzonych sa w `data/cable/ground_truth/<klasa>/*_mask.png`

Do przygotowania przykladowego rozwiazania:

1. Zbierz razem wszystkie dostepne obrazy `good` i wszystkie obrazy uszkodzone.
2. Zbuduj wlasny, deterministyczny podzial `train/val` na wymieszanym zbiorze.
3. Walidacje wykonuj tylko na swoim podziale, a nie na oryginalnym ukladzie katalogow.

### Co ma powstac

Przygotuj cale, przykladowe rozwiazanie obejmujace:

1. Wczytanie i przygotowanie danych.
2. Wstepne przetwarzanie obrazu.
3. Model / detektor do segmentacji binarnej wad.
4. Ewaluacje na wlasnym zbiorze walidacyjnym.
5. Gotowy plik `sample_submission/model.py`, ktory dziala samodzielnie i implementuje `predict(image)`.
6. `requirements.txt` z potrzebnymi bibliotekami.

### Wymagania techniczne

- Srodowisko docelowe: Python `3.12`
- Dostepne biblioteki w runtime submission: `numpy`, `Pillow`
- Limit czasu wykonania pojedynczego zgłoszenia: `300 s`
- Rozwiazanie ma byc lekkie i uruchamialne bez trenowania w czasie inferencji

### Oczekiwany styl rozwiazania

Przykladowe rozwiazanie ma byc praktyczne, proste do uruchomienia i dobrze opisane w kodzie. Nie trzeba budowac modelu SOTA. Wazne, zeby:

- pipeline byl spojny,
- split danych byl poprawny,
- metryka byla policzona,
- `sample_submission/model.py` byl samowystarczalny,
- kod byl czytelny i mozliwy do dalszego rozwijania.

### Metryka

Uzyj sredniego IoU:

- dla obrazow uszkodzonych porownaj predykcje z maska referencyjna,
- dla obrazow `good` maska referencyjna jest zerowa.

### Dodatkowe zalecenia

- Nie hard-code'uj wynikow pod konkretne obrazy testowe.
- Jesli budujesz rozwiazanie regułowe lub wzorcowe, opisz jasno logike.
- Jesli stroisz progi, rob to na walidacji zbudowanej po wymieszaniu danych.
- Dopilnuj, aby finalny `predict()` zwracal tylko `uint8` oraz wartosci `0/255`.
