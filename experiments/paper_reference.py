"""Published reference data: Table 2 of Jiang & Zhang, Performance Evaluation 60 (2005) 5-29.

THESE ARE THE PAPER'S MEASUREMENTS (Linux 2.2.14, Pentium II 400 MHz, 384 MB, real SPEC2000 etc.).
They are reference data only -- never produced by this repository's code.

All values were checked against Table 2 on page 26 of the paper PDF (the project brief omitted the
vortex/gcc column; its values are in the paper: LRU 248,045/275,982, token 51,749/12,425).
Note: the paper's column heading is "vortex/gcc" (Sec. 4.2 group 3 is vortex3 + gcc).

Per-process order follows the paper's column headings:
  gzip/vortex, bit-r/gcc, vortex/gcc (vortex first), vortex-1/vortex-3, LU-1/LU-2.
"""

PAPER_TABLE2 = {
    "gzip_vortex3":    dict(names=("gzip", "vortex3"),
                            lru_pf=(328_551, 294_038), tok_pf=(179_908, 59_945), pf_red=(45, 80),
                            lru_sd=(5.23, 3.85), tok_sd=(2.63, 1.83), sd_red=(50, 52)),
    "bitr_gcc":        dict(names=("bit-r", "gcc"),
                            lru_pf=(108_352, 139_807), tok_pf=(86_467, 25_406), pf_red=(20, 82),
                            lru_sd=(2.69, 3.63), tok_sd=(2.08, 2.25), sd_red=(23, 38)),
    "vortex3_gcc":     dict(names=("vortex3", "gcc"),
                            lru_pf=(248_045, 275_982), tok_pf=(51_749, 12_425), pf_red=(79, 95),
                            lru_sd=(3.37, 5.61), tok_sd=(1.54, 1.85), sd_red=(54, 67)),
    "vortex1_vortex3": dict(names=("vortex1", "vortex3"),
                            lru_pf=(193_399, 208_342), tok_pf=(13_842, 76_171), pf_red=(93, 63),
                            lru_sd=(3.58, 3.33), tok_sd=(1.95, 2.08), sd_red=(46, 38)),
    "lu1_lu2":         dict(names=("LU-1", "LU-2"),
                            lru_pf=(138_372, 77_680), tok_pf=(17_999, 168_215), pf_red=(87, -117),
                            lru_sd=(3.57, 3.40), tok_sd=(2.57, 2.99), sd_red=(28, 12)),
}

# Source note: Table 2 prints the LU-2 page-fault reduction as (-117)%; the running text of Sec. 6 says -116%.
# 100*(77,680-168,215)/77,680 = -116.55%, so both roundings are defensible.
