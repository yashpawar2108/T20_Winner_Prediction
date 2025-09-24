import batting_preprocessing
import bowling_preprocessing
import feature_engineering
import visualization

def main():
    print("Starting batting preprocessing...")
    batting_preprocessing.process_batting_data()
    print("Batting preprocessing finished.")

    print("Starting bowling preprocessing...")
    bowling_preprocessing.process_bowling_data()
    print("Bowling preprocessing finished.")

    print("Starting feature engineering...")
    feature_engineering.create_features()
    print("Feature engineering finished.")

    print("To view the interactive dashboard, run the following command in your terminal:")
    print("streamlit run dashboard.py")

if __name__ == '__main__':
    main()
