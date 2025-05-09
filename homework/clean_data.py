import pandas as pd
import string
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def load_data(input_file):
    df = pd.read_csv(input_file)
    return df


def create_key(df):
    df_copy = df.copy()
    # Normalizar: quitar espacios y pasar a minúsculas
    df_copy['key'] = df_copy['raw_text'].str.strip().str.lower()
    # Eliminar puntuación
    df_copy['key'] = df_copy['key'].str.translate(str.maketrans('', '', string.punctuation))
    # Dividir en tokens
    df_copy['key'] = df_copy['key'].str.split()
    # Aplicar Porter Stemmer, eliminar duplicados y ordenar
    df_copy['key'] = df_copy['key'].apply(
        lambda tokens: sorted({stemmer.stem(tok) for tok in tokens if tok})
    )
    # Unir los stems con espacio
    df_copy['key'] = df_copy['key'].apply(lambda stems: ' '.join(stems))
    return df_copy


def generate_cleaned_column(df):
    df2 = df.copy()
    # Mapea cada key a la raw_text con mayor frecuencia
    most_common = (
        df2.groupby('key')['raw_text']
        .agg(lambda x: x.value_counts().idxmax())
    )
    df2['cleaned_text'] = df2['key'].map(most_common)
    return df2


def save_data(df, output_file):
    out = df[['raw_text', 'cleaned_text']].copy()
    out['cleaned_text'] = out['cleaned_text'].str.translate(
        str.maketrans('', '', string.punctuation)
    )
    out.to_csv(output_file, index=False)


def main(input_file, output_file):
    df = load_data(input_file)
    df = create_key(df)
    df = generate_cleaned_column(df)
    # Generar el CSV que revisa el test
    df.to_csv('files/test.csv', index=False)
    # Guardar la salida final
    save_data(df, output_file)


if __name__ == '__main__':
    main('files/input.txt', 'files/output.txt')
