# 🛰️ Mars Autonomous Landing Site Selection

This project simulates intelligent Mars landing site selection using DEM data, slope, soil hardness, rock obstacles, and distance to a scientific target. It computes terrain suitability using a weighted scoring system, highlights safe zones, and identifies the best autonomous landing point.

## 🚀 Features
- Supports three modes:
  - **Baseline**: Raw DEM input
  - **After Training**: Fine-tuned with HiRISE/Curiosity data
  - **Previous Missions**: NASA historical datasets
- Calculates slope, soil hardness, rock hazard maps, and target proximity
- Scores each pixel based on:
  - Slope (40%)
  - Soil Hardness (30%)
  - Rocks (20%)
  - Distance (10%)
- Classifies safe zones and selects the most suitable landing point
- Outputs high-resolution visualizations (`.png`, `.svg`) and a performance metrics table (`.csv`)

## 📁 Folder Structure
```
mars-landing-simulator/
├── message.txt                # Python simulation code
├── data/                      # Input DEM files (.tif)
├── outputs/                   # Plots and safe zone maps
├── model_comparison_results.csv
└── README.md
```

## ⚙️ Dependencies
Install with:
```bash
pip install numpy matplotlib rasterio scipy psutil pandas
```

## ▶️ How to Run
1. Add the required `.tif` files to `/data`
2. Run the script:
```bash
python message.txt
```
3. View:
   - Best landing point
   - Safe zone maps
   - `model_comparison_results.csv`

## 📊 Example Output
- Accuracy: 94.6%
- Hazard Detection: 96.8%
- Landing Error: ~20 m
- Latency: 0.48s
- Fuel Efficiency: 85%
