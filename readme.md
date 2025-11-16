# PS3 - Nonparametric Estimation of First Price Procurement Auction

This project implements the **GPV (Guerre, Perrigne, Vuong) two-step nonparametric estimation** method for analyzing first-price procurement auctions using real data from Colorado Department of Transportation (CDOT) bid tabs.

## Project Structure

```
PS3/
├── data/
│   ├── bid_min_and_count.csv      # Processed auction data (t, N, W)
│   ├── cdot_bid_tabs_index.csv    # Index of scraped PDFs
│   ├── FC_steps.csv               # Estimated cost CDFs
│   ├── FC_steps_plot.pdf          # Visualization of cost distributions
│   └── pdf/                       # Downloaded CDOT bid tab PDFs
├── src/
│   ├── scrape.py                  # Web scraper for CDOT bid tabs
│   ├── data.py                    # PDF parser to extract bid data
│   └── calculation.py             # GPV estimation implementation
├── docs/
│   └── GPVprocurement.ipynb       # Jupyter notebook with analysis
├── requirements.txt               # Python dependencies
└── readme.md                      # This file
```

## Overview

The project consists of three main components:

### 1. Data Collection (`src/scrape.py`)
- Scrapes CDOT bid tab PDFs from their archive pages
- Uses Playwright for browser automation to handle dynamic content
- Downloads PDFs and creates an index of available bid tabs

### 2. Data Processing (`src/data.py`)
- Parses PDF bid tables using `pdfplumber`
- Extracts:
  - `t`: Auction identifier (file path)
  - `N`: Number of bidders
  - `W`: Winning (lowest) bid
- Outputs cleaned data to `bid_min_and_count.csv`

### 3. GPV Estimation (`src/calculation.py` & `docs/GPVprocurement.ipynb`)
- Implements the GPV two-step nonparametric estimation method
- Analyzes auctions at different competition levels (25th, 50th, 75th percentiles of N)
- Estimates the distribution of bidders' costs from observed winning bids



## Usage

### Step 1: Scrape CDOT Bid Tabs
```bash
python -m src.scrape
```
This downloads PDF bid tabs to `data/pdf/` and creates an index in `data/cdot_bid_tabs_index.csv`.

### Step 2: Parse PDFs
```bash
python -m src.data --pdf-dir data/pdf --out data/bid_min_and_count.csv
```
This extracts auction data from PDFs and creates the main dataset.

### Step 3: Run GPV Estimation
```bash
python -m src.calculation --data data/bid_min_and_count.csv --out-csv data/FC_steps.csv --out-plot data/FC_steps_plot.pdf
```
This performs the GPV estimation and generates:
- `FC_steps.csv`: Estimated cost distribution functions
- `FC_steps_plot.pdf`: Visualization of the results


## Data Description

### Input Data (`bid_min_and_count.csv`)
| Column | Description |
|--------|-------------|
| `t` | Auction identifier (PDF file path) |
| `N` | Number of bidders participating in the auction |
| `W` | Winning bid (lowest bid in procurement auction) |

### Output Data (`FC_steps.csv`)
| Column | Description |
|--------|-------------|
| `level` | Competition level (low/median/high) |
| `N_label` | Number of bidders at this level |
| `N_used` | Actual N values used (may include pooling) |
| `c` | Estimated cost values |
| `FC` | Cumulative distribution function F(c) |

## Methodology

The GPV method estimates the distribution of private costs from observed bids in first-price auctions:

1. **Step 1**: Estimate the distribution of winning bids F_W(w) and density f_W(w)
2. **Step 2**: Transform winning bids to pseudo-costs using the GPV inversion formula:
   ```
   c = w - (1/(N-1)) × (1-F_W(w))/f_W(w)
   ```
3. **Step 3**: Estimate the distribution of costs F_C(c) from pseudo-costs
4. **Analysis**: Compare cost distributions across different competition levels

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Matplotlib (visualization)
- Statsmodels (empirical distributions, KDE)
- pdfplumber (PDF parsing)
- requests, BeautifulSoup4 (web scraping)
- Playwright (browser automation)

See `requirements.txt` for complete dependencies.

## Notes

- The dataset focuses on Colorado Department of Transportation procurement auctions
- Analysis uses percentiles (25th, 50th, 75th) of N to represent low, medium, and high competition
- The notebook includes data filtering to handle edge cases (N ≤ 2, NaN values)


## References

- Guerre, E., Perrigne, I., & Vuong, Q. (2000). "Optimal Nonparametric Estimation of First-Price Auctions." *Econometrica*, 68(3), 525-574.


