# Berbagi Pengetahuan Lintas Kelompok Pascaperubahan Institusional: Simulasi Teori Permainan Evolusioner

Kode simulasi dan data hasil untuk makalah IKMS 2026 (Indonesia Knowledge Management Summit):

> H. Alphionanda, A. C. Ramadhani, dan S. M. Albayani, "Kerangka Teori Permainan Evolusioner untuk Berbagi Pengetahuan Lintas Kelompok Pascaperubahan Institusional: Kasus Ilustratif Penggabungan PT Pertamina Patra Niaga," *Prosiding IKMS 2026*, Institut Teknologi Bandung, 2026.

*English summary.* Agent-based evolutionary game model of knowledge sharing across groups after a merger. 300 agents in three communities (stochastic block model) choose among four sharing strategies that spread by Fermi imitation; in-group bias is an extra cost of sharing across groups. The code tests four policy instruments (bonus, forum, job rotation, reputation), their sequencing, and network integration effects on collective solution quality (NK landscape). All results come from simulation, not from employee data; the Pertamina Patra Niaga merger is an illustrative case.

## Model singkat

- Tiga kelompok asal, masing-masing 100 agen; rata-rata 4 tautan dalam kelompok dan 1 tautan lintas kelompok per agen.
- Empat strategi: H (menahan), I (hanya ke kelompok sendiri), O (hanya ke kelompok lain), S (ke semua).
- Setiap tautan adalah permainan donasi: penerima mendapat manfaat *b*, pemberi menanggung biaya *c*. Biaya ke kelompok sendiri c(1 − φ), ke kelompok lain c(1 + h)(1 − φ) − β, ditambah imbalan reputasi ρb per tindakan berbagi lintas kelompok.
- Agen meniru tetangga acak dengan peluang Fermi 1 / (1 + exp((U_i − U_j)/K)), ditambah eksplorasi acak dengan peluang μ.
- Parameter utama: b/c = 12, c = 1, K = 0,5, μ = 0,001, 3.000 MCS, hasil dirata-rata pada 1.000 MCS terakhir.
- Lapis kedua: pola berbagi akhir menjadi jalur penyalinan solusi pada *NK landscape* (N = 15, K_NK = 5 dan 10).

## Cara menjalankan

Python 3.10 atau lebih baru.

```bash
pip install -r requirements.txt
python baseline_lattice.py 100 5000   # validasi mesin imitasi terhadap Szabó dkk. (2005), sekitar 2 menit
python run_v2.py                      # kurva bias, instrumen, belanja, sensitivitas, sapuan b/c (sekitar 10 menit, 2 inti)
python make_figures_v2.py
python run_v3.py                      # NK dengan lanskap khas kelompok (λ) dan daya serap (α)
python run_v31.py && python analyze_v31.py   # eksperimen urutan kebijakan, 128 ulangan, uji permutasi dan Holm
python run_v32.py                     # peta kebijakan h × β
python run_v33_traj.py                # lintasan waktu melewati rotasi, 128 ulangan
python make_figures_v32.py
python make_figures_v3.py
cd poster && python make_figs_poster.py   # gambar poster dari hasil di figures/
```

Semua hasil (JSON dan PNG) ditulis ke `figures/`. Hasil yang dipakai di makalah sudah disertakan, jadi gambar dapat dibuat ulang tanpa menjalankan simulasi. Bibit acak tetap: menjalankan ulang memberi angka yang sama.

## Dari angka di makalah ke berkas

| Hasil di makalah | Skrip | Data |
| --- | --- | --- |
| Validasi kisi persegi (Szabó dkk. 2005) | `baseline_lattice.py` | `figures/baseline_results.json` |
| Pengaruh bias *h* terhadap berbagi dan peluang tidak silo | `run_v2.py`, `make_figures_v2.py` | `figures/v2_results.json`, `v2_derived.json` |
| Pengaruh instrumen, belanja setara, uji sensitivitas, sapuan *b/c* | `run_v2.py`, `make_figures_v2.py` | `figures/v2_results.json` |
| Kualitas solusi pada NK landscape (λ, α) | `run_v3.py` | `figures/v3_results.json` |
| Urutan kebijakan dan pencabutan insentif (uji selisih, koreksi Holm) | `run_v31.py`, `analyze_v31.py` | `figures/v31_results.json` |
| Peta kebijakan *h* × β, potret jaringan | `run_v32.py`, `make_figures_v32.py` | `figures/v32_results.json` |
| Lintasan waktu melewati rotasi | `run_v33_traj.py`, `make_figures_v32.py` | `figures/v33_results.json` |
| Peta efek rotasi terhadap bias (tambahan untuk poster) | `run_v34_rotmap.py` | `figures/v34_rotmap.json` |
| Gambar poster | `poster/make_figs_poster.py` | membaca JSON di `figures/` |

`model.py` berisi jaringan, utilitas, dan dinamika dasar; `model_v1.py` menambahkan status akhir strategi, belanja kebijakan, dan lapis NK.

Berkas `run_v0.py`, `run_v1.py`, `make_figures_v1.py`, beserta `figures/v0_*` dan `figures/v1_*`, adalah versi awal yang disimpan sebagai arsip dan tidak dipakai untuk angka makalah.

## Keterbatasan

Nilai *h*, *b/c*, struktur jaringan, dan aturan imitasi adalah asumsi model. Kalibrasi *h* dari ukuran efek metaanalisis Balliet dkk. (2014) bersifat heuristik (*h* ≈ 0,32–0,43); skenario *h* = 0,6 dan 1,0 adalah uji tekanan. Pemetaan instrumen ke program Patra Niaga hanya ilustrasi.

## Lisensi

Kode: MIT (`LICENSE`). Data hasil dan gambar di `figures/` dan `poster/`: CC BY 4.0 (`LICENSE-DATA`). `poster/publication.mplstyle` berasal dari K-Dense scientific-agent-skills (MIT).

Bila memakai kode atau data ini, mohon sitasi makalah di atas (lihat `CITATION.cff`).

## Kontak

Heart Alphionanda, Institut Teknologi Bandung: 13623022@mahasiswa.itb.ac.id
