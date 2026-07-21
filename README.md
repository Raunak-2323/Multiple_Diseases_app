# Multiple_Diseases_app
https://github.com/user-attachments/assets/f3bcd7bf-347b-4912-a432-bda9a41611c2
                    ┌─────────────────────┐
                    │   Healthcare Datasets │
                    │ (Diabetes, Heart,    │
                    │  Parkinson's)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ • Cleaning          │
                    │ • Feature Selection │
                    │ • Data Formatting   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train-Test Split    │
                    │      (80:20)        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest       │
                    │ Model Training      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ & Serialization     │
                    │   (.pkl files)      │
                    └──────────┬──────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────┐
│                 Streamlit Web App                    │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ User Inputs       │
              │ Medical Features  │
              │ (40+ Parameters)  │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Disease Selection │
              │ Diabetes / Heart  │
              │ / Parkinson's     │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Load Corresponding│
              │ Random Forest     │
              │ Model (.pkl)      │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Real-Time         │
              │ Prediction        │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Disease Risk      │
              │ Result Displayed  │
              └───────────────────┘
