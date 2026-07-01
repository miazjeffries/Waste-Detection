""" 
    Splits the deduplicated, unified data into training, validation, and testing sets with a stratified split 
    After deduplication, before resampling for class balance
"""

import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = 'data/processed/deduplicated_data.csv'
output_csv = 'data/processed/split_data.csv'

# Split ratio: 80% train, 10% validation, 10% test
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def main():
    df = pd.read_csv(input_csv)

    # Split 80% train, 20% temporary hold
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=42
    )

    temp_df = temp_df.reset_index(drop=True)

    # Split 20% temp into half (10% val, 10% test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio) / (val_ratio + test_ratio),
        stratify=temp_df['label'],
        random_state=42
    )

    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Combine back together
    df_split = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_split.to_csv(output_csv, index=False)

    print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val : {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Show class distribution per split
    for split in ['train', 'val', 'test']:
        split_data = df_split[df_split['split'] == split]
        print(f"\n{split} class distribution: ")
        print(split_data['label'].value_counts().sort_index())
    
    print(f"\nSaved to {output_csv}")


if __name__ == "__main__":
    main()