# Analysis Results: Sounds of Home

**Subset:** Day 14 only

**Dataset source:** [Sounds of Home](https://www.cvssp.org/data/ai4s/sounds_of_home/)

**Analysis generated:** 2026-10-21 11:46:55 IST

## Results from the Latest Run

### 1. Analysis Settings

- **Dataset directory:** `data/Raw_data`
- **Audio files discovered:** 28
- **Prediction probability threshold:** 0.2
- **Audio block size:** 10,240 samples
- **Blocks sampled per audio file:** Up to 400

### 2. Per-File Results

| Recording | Estimated erased blocks | Prediction frames | Top 4 classes above threshold |
|---|---:|---:|---|
| 20231116_100000 | 55.2% | 16,852 | Music (11,753), Television (6,795), Inside, small room (6,526), Speech (4,443) |
| 20231116_110000 | 51.2% | 16,852 | Music (12,474), Television (7,362), Inside, small room (6,644), Speech (5,181) |
| 20231116_180000 | 42.2% | 16,852 | Inside, small room (7,070), Speech (6,975), Television (6,734), Music (3,570) |
| 20231116_200000 | 44.8% | 16,852 | Speech (9,202), Inside, small room (8,214), Television (6,751), Music (3,238) |
| 20231116_210000 | 43.5% | 16,852 | Speech (8,086), Television (7,229), Inside, small room (6,373), Music (4,349) |
| 20231117_080000 | 0.2% | 16,852 | Silence (9,466), Inside, small room (4,934), Static (4,385), White noise (3,186) |
| 20231117_160000 | 1.5% | 16,852 | Silence (9,435), Static (4,845), Inside, small room (4,799), Drawer open or close (2,903) |
| 20231117_180000 | 33.8% | 16,852 | Speech (6,921), Silence (5,704), Inside, small room (5,521), Television (5,181) |
| 20231117_190000 | 51.5% | 16,852 | Speech (13,416), Television (9,363), Inside, small room (6,595), Male speech, man speaking (4,258) |
| 20231117_200000 | 39.0% | 16,852 | Speech (8,185), Inside, small room (7,299), Television (5,794), Animal (2,700) |
| 20231117_220000 | 11.5% | 16,852 | Silence (11,223), Static (6,839), Inside, small room (2,700), White noise (2,476) |
| 20231118_100000 | 3.2% | 16,852 | Silence (8,293), Inside, small room (5,047), Static (2,751), White noise (2,678) |
| 20231119_130000 | 7.0% | 16,852 | Silence (5,151), Inside, small room (4,245), Bee, wasp, etc. (4,182), Insect (4,025) |
| 20231120_070000 | 0.0% | 16,852 | Silence (14,919), Static (8,674), Inside, small room (3,905), White noise (2,193) |
| 20231120_120000 | 48.8% | 16,852 | Music (9,495), Inside, small room (6,604), Television (5,940), Speech (5,762) |
| 20231120_140000 | 51.5% | 16,852 | Music (10,952), Television (7,232), Inside, small room (5,798), Speech (5,080) |
| 20231120_180000 | 52.0% | 16,852 | Speech (10,996), Television (10,852), Music (7,625), Inside, small room (5,241) |
| 20231120_200000 | 23.8% | 16,852 | Inside, small room (7,678), Silence (4,627), Speech (3,496), Television (3,457) |
| 20231121_080000 | 0.2% | 16,852 | Silence (14,060), Static (9,141), Inside, small room (3,128), White noise (2,143) |
| 20231121_100000 | 52.5% | 16,852 | Speech (10,067), Television (9,464), Music (8,290), Inside, small room (7,460) |
| 20231121_130000 | 40.5% | 16,852 | Speech (7,954), Inside, small room (6,613), Music (6,031), Television (5,820) |
| 20231121_210000 | 20.8% | 16,852 | Inside, small room (7,214), Silence (6,618), Speech (4,153), Static (3,896) |
| 20231122_090000 | 0.8% | 16,852 | Silence (11,068), Static (7,413), Inside, small room (4,771), White noise (2,970) |
| 20231122_100000 | 2.8% | 16,852 | Silence (15,657), Static (12,052), White noise (5,255), Inside, small room (1,576) |
| 20231122_110000 | 8.2% | 16,852 | Inside, small room (8,121), Silence (5,349), Drawer open or close (3,708), Snort (3,374) |
| 20231122_140000 | 19.2% | 16,852 | Silence (9,105), Static (7,180), Inside, small room (4,865), Speech (4,525) |
| 20231122_190000 | 30.0% | 16,852 | Speech (5,407), Inside, small room (3,938), Vehicle (3,883), Television (3,826) |
| 20231122_200000 | 20.2% | 16,852 | Vehicle (8,118), White noise (7,041), Rumble (4,248), Speech (2,523) |

### 3. Overall Summary

- **Recordings analyzed successfully:** 28
- **Total prediction frames:** 471,856
- **Estimated analyzed duration:** 27.96 hours

### 4. Most Frequently Predicted Sound Classes

The table below counts prediction-frame occurrences where a class has a probability of at least 0.2. These are counts, not unique sound events.

| Rank | Sound class | Frames above threshold | Percentage of all prediction frames |
|---:|---|---:|---:|
| 1 | Inside, small room | 154,071 | 32.7% |
| 2 | Silence | 143,945 | 30.5% |
| 3 | Speech | 128,697 | 27.3% |
| 4 | Television | 110,567 | 23.4% |
| 5 | Music | 91,598 | 19.4% |
| 6 | Static | 91,112 | 19.3% |
| 7 | Animal | 57,333 | 12.2% |
| 8 | White noise | 49,868 | 10.6% |
| 9 | Snort | 44,834 | 9.5% |
| 10 | Domestic animals, pets | 41,301 | 8.8% |
| 11 | Buzz | 37,355 | 7.9% |
| 12 | Breathing | 31,181 | 6.6% |
| 13 | Writing | 30,746 | 6.5% |
| 14 | Drawer open or close | 28,404 | 6.0% |
| 15 | Male speech, man speaking | 24,755 | 5.2% |
| 16 | Snake | 24,240 | 5.1% |
| 17 | Tick-tock | 21,673 | 4.6% |
| 18 | Mouse | 20,719 | 4.4% |
| 19 | Sliding door | 18,153 | 3.8% |
| 20 | Narration, monologue | 16,135 | 3.4% |
| 21 | Vehicle | 15,378 | 3.3% |
| 22 | Tick | 14,331 | 3.0% |
| 23 | Snoring | 13,655 | 2.9% |
| 24 | Hum | 13,357 | 2.8% |
| 25 | Mechanical fan | 12,541 | 2.7% |
| 26 | Printer | 11,803 | 2.5% |
| 27 | Dog | 11,426 | 2.4% |
| 28 | Cat | 11,197 | 2.4% |
| 29 | Musical instrument | 10,939 | 2.3% |
| 30 | Door | 9,332 | 2.0% |

### 5. Notes and Limitations

- **Erased audio estimate:** A sampled block is classified as erased when more than 99% of its samples are exactly zero. The reported percentage is the fraction of sampled blocks meeting this condition, not the exact percentage of audio duration erased.
- **Prediction threshold:** Only class predictions with probability at least 0.2 are counted.
- **Frame counts are not event counts:** A sound lasting several seconds may contribute many prediction frames.
- **Model predictions are not ground truth:** The reported classes are model-generated annotations and may include false positives or miss real sounds.
- **Duration estimate:** Total hours are calculated from the number of prediction frames and the configured block size. Treat this as an approximate estimate of analyzed duration.

---
