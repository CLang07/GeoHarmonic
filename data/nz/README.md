# NZ Well Correlation – Maui/Tui area, offshore Taranaki

Source: NZP&M Geodata Catalogue (geodata.nzpam.govt.nz), "Wireline Data" ZIPs, downloaded 22 Nov 2026.
All depths are metres MD as delivered.

## Layout
- `wells/<Well>/las/` – main open-hole LAS (use these for correlation)
- `wells/<Well>/dipmeter/` – SHDT dipmeter LAS (not needed for correlation)
- `wells/<Well>/checkshot/` – checkshot / VSP data (for later well ties)
- `wells/<Well>/mudlog/` – lithology descriptions, comments, deviation survey
- `wells/<Well>/dlis/` – DLIS originals where provided
- `reports/PR3099/` – Tui-1 vs Maui palynology correlation (answer key) + 2 enclosures
- `curve_inventory_summary.csv` – key curve coverage per well
- `curve_inventory_detail.csv` – every GR/RES/RHOB/NPHI/DT/PE/SP/CAL curve by file

## Notes / gotchas
- Maui-1 and Maui-3 were delivered as `.dat`; they are LAS 2.0 and were renamed `.las`.
- Maui-4 only covers 1311–2316 m (no Kapuni F-sand section).
- Kiwi-1 is LWD only, 2917–4229 m (reservoir section); no sonic.
- Maui-1/3/4 have no neutron (Maui-3/4 have SNP sidewall neutron, not classified as NPHI).
- Maui-7 `LDL-CNL-NGS_2.LAS` header says WELL = "TE RANGA # 1" – check before use.
- Many Schlumberger files contain main + repeat passes in one file (duplicate mnemonics). The tool needs to pick the main pass.
- Pateke-2 logs are DLIS only (`Pateke_2_DLIS_Final_Combine.dlis`, 115 MB). Not copied here – still in
  `Downloads\collection\nz-pateke-2-all\pateke-2.zip`. Convert with `pip install dlisio`.
- Skipped: production/well-test LAS (Maui-7 well cleaning, multi-rate test), sonic waveform DLIS, 5-second drilling data.
- Still to download: Tui-1 (600 MB) and Amokura-1 (1.62 GB).
