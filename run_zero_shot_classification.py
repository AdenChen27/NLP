import pandas as pd
from transformers import pipeline


# labels for zero shot classification
relevancy_labels = [
    "This is an article about superintendents", 
    "This is NOT an article about superintendents", 
]

topic_labels = [
    "Education Policy", 
    "Personal Life", 
    "Budget and Funding", 
    "School Safety", 
    "Community Engagement", 
    "Curriculum Updates", 
    "Technology Integration", 
    "Student Achievement", 
    "Equity and Inclusion", 
    "Infrastructure and Facilities", 
    "Government Interaction"
]
topic_labels = ["This article is about " + l for l in topic_labels]



def f_zero_shot_classification(df, labels, out_filename=None, *args, **kwargs):
    """
    Given labels, perform zero-shot-classification for 
        the "Article Text" row in df; save results in df

    - `df` can be filenames or dataframes
    - if provided, will save output also to `out_filename`
    """
    if type(df) is str:
        df = pd.read_csv(df)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda")

    # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device_map="auto")
    # classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-9", device=0)

    for index, row in df.iterrows():
        text = str(row["Article Text"])
        result = classifier(text, labels, truncation=True)

        for i, label in enumerate(labels):
            # df_label: column name in df
            df_label = "ZSC-" + label.replace("This article is about ", "").replace(" ", "-")
            df.loc[index, df_label] = result["scores"][i]

    if df is not None:
        df.to_csv(out_filename, index=False)

    return df


if __name__ == '__main__':
    f_zero_shot_classification("data/df-processed.csv", relevancy_labels, "data/df-processed.csv")
    f_zero_shot_classification("data/df-processed.csv", topic_labels, "data/df-processed.csv")



