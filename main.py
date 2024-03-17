from src.dataset import EngagementDataset


def main():
    enagement_score_data = EngagementDataset()
    print(enagement_score_data)
    print(enagement_score_data.primary_key_column)
    print(enagement_score_data.timestamp_column)


if __name__ == "__main__":
    main()
