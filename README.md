# Attrition Radar

A proactive Attrition Intelligence Dashboard that uses internal signals to predict and prevent talent loss before it happens.

## 🎯 Overview

Attrition Radar helps organizations predict and prevent talent loss by:
- Identifying teams and departments at risk of losing talent
- Suggesting roles to pre-source before attrition occurs
- Visualizing patterns of churn across different dimensions
- Providing actionable insights for strategic hiring

## 🚀 Features

- **Attrition Risk Prediction**: Machine learning models to identify high-risk employees and teams
- **Pre-hiring Recommendations**: Smart suggestions for roles to start sourcing
- **Interactive Dashboard**: Visualize attrition patterns and trends
- **Risk Heatmaps**: Department and region-based risk visualization
- **Attrition Timeline**: Historical trends and future predictions

## 📊 Data Sources

The system integrates multiple data sources:
- HRIS data (employee records, roles, tenure, performance)
- Attrition logs (exit reasons, feedback)
- Productivity metrics (optional)
- Industry benchmarks

## 🛠️ Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your data sources in `config.yaml`
4. Run the data pipeline:
```bash
python src/data/pipeline.py
```

5. Start the dashboard:
```bash
python src/app.py
```

## 📁 Project Structure

```
attrition_radar/
├── src/
│   ├── data/           # Data processing and feature engineering
│   ├── models/         # ML models for attrition prediction
│   ├── visualization/  # Dashboard and visualization components
│   └── utils/          # Helper functions and utilities
├── tests/              # Unit tests
├── config/             # Configuration files
└── notebooks/          # Jupyter notebooks for analysis
```

## 🔧 Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## 📈 Usage

1. **Data Ingestion**: Import your HRIS and attrition data
2. **Model Training**: Train the attrition prediction model
3. **Dashboard**: Access insights through the interactive dashboard
4. **Alerts**: Set up notifications for high-risk situations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 