import matplotlib.pyplot as plt
import seaborn as sns

def plot_survival_by_family(df):
    survival_rates = df.groupby('FamilySize')['Survived'].mean()

    survival_rates.plot(kind='bar')

    plt.title('Survival Rate by Family Size')
    plt.xlabel('Family Size')
    plt.ylabel('Survival Rate')

    plt.show()

def plot_survival_by_sex(df):

    survival_rates = df.groupby('Sex')['Survived'].mean()

    survival_rates.plot(kind='bar')

    plt.title('Survival Rate by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Survival Rate')

    plt.show()

def plot_survival_by_class(df):

    survival_rates = df.groupby('Pclass')['Survived'].mean()

    survival_rates.plot(kind='bar')

    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Survival Rate')

    plt.show()

def plot_age_distribution(df):

    sns.histplot(df['Age'], kde=True)

    plt.title('Age Distribution')

    plt.xlabel('Age')
    plt.ylabel('Count')

    plt.show()

def plot_age_boxplot(df):

    sns.boxplot(x=df['Age'])

    plt.title('Age Boxplot')

    plt.show()

def plot_correlation_heatmap(df):

    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm'
    )

    plt.title('Correlation Heatmap')

    plt.show()