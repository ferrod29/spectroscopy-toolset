# Example data

| File | Content |
|------|---------|
| `TestData_1.dat` | Transient-absorption matrix (2068 wavelengths x 576 delays) from the Pharos-based TA setup. |
| `TestData_2.dat` | A second TA matrix with the same layout, recorded on a delay grid offset by +1.9 ps. It is a different measurement, not a repeat of `TestData_1.dat`, so the two must not be averaged. |
| `shotoshot.txt` | Two-column data kept from the original repository (x from -170 to 170). The name suggests a shot-to-shot noise measurement. It loads with `read_spectrum`. |

## TA matrix layout

```
0        t_1      t_2      ...   t_M        <- delays (ps; raw delay-stage positions)
lambda_1 dA_11    dA_12    ...   dA_1M
lambda_2 dA_21    ...
...                                          <- wavelengths (nm), dA in OD
```

Some properties of `TestData_1.dat` that the examples rely on:

* The delays are raw stage positions (-285.5 to -85.5 ps). Time zero lies about 0.55–0.9 ps
  after the first delay, depending on wavelength, because of the probe chirp.
* The pump at 515 nm scatters into roughly 512–519 nm, where the values are saturated.
  Remove that band with `exclude_wavelengths((509, 523))`.
* The main feature is a long-lived ground-state bleach around 500–600 nm. A coherent
  artefact (cross-phase modulation) appears around time zero.
