#   Imports:
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from managed_db import *

feature_name_best = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time','DEATH_EVENT']

gender_dict = {"Bărbat": 1, "Femeie": 0}
feature_dict = {"Da": 1, "Nu": 0}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key

def get_feature_value(val):
    feature_dict = {"Da": 1, "Nu": 0}
    for key, value in feature_dict.items():
        if val == key:
            return value

def load_model(model_file):
    """Funcție pentru a incarca modelul"""
    loaded_model = joblib.load(os.path.join(model_file))
    return loaded_model


def main():

    menu = ['Acasă', 'Analiza setului de date', 'Diagnosticare', 'Feedback']
    submenu = ['Verificarea datelor', 'Distribuția atributelor', 'Distribuții comparative cu probabilitatea de moarte', 'Diagrame de tip "pie"']

    choice = st.sidebar.selectbox('Meniu', menu)

    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    if choice == 'Acasă':
        st.title('Aplicație web pentru diagnosticarea insuficienței cardiace')
        st.header('Diagnosticarea insuficienței cardiace prin utilizarea tehnologiei Machine Learning')

        image = Image.open('541-5413272_insurance-and-medical-cost-clipart.png')
        st.image(image, use_column_width=True)

    elif choice == 'Analiza setului de date':
        st.title('Analiza setului de date')
        option = st.selectbox('Opțiuni', submenu)

        if option == "Verificarea datelor":
            st.header("Verificarea corectitudinii datelor")
            st.subheader('Întreaga bază de date folosită')
            st.dataframe(df)

            st.subheader('Verificarea faptului că nu avem date nule')
            st.write(df.isnull().sum())

        if option == "Distribuția atributelor":
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['age'],
                xbins = dict(
                    start=40,
                    end=95,
                    size=2
                ),
                marker_color='#e8ab60',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția vârstei',
                xaxis_title_text='Vârstă',
                yaxis_title_text='Număr',
                bargap=0.05,
            )

            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['creatinine_phosphokinase'],
                xbins = dict(
                    start=23,
                    end=582,
                    size=15
                ),
                marker_color='#fe6f5e',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția enzimelor CPK',
                xaxis_title_text='Enzime CPK',
                yaxis_title_text='Număr',
                bargap=0.05,
            )
            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['ejection_fraction'],
                xbins = dict(
                    start=14,
                    end=80,
                    size=2
                ),
                marker_color='#a7f432',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția fracțiunii de ieșire a sângelui',
                xaxis_title_text='Fracțiunea de ieșire',
                yaxis_title_text='Număr',
                bargap=0.05,
            )
            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['platelets'],
                xbins = dict(
                    start=25000,
                    end=300000,
                    size=5000
                ),
                marker_color='#50bfe6',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția trombocitelor',
                xaxis_title_text='Trombocite',
                yaxis_title_text='Număr',
                bargap=0.05,
            )
            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['serum_creatinine'],
                xbins = dict(
                    start=0.5,
                    end=9.4,
                    size=0.2
                ),
                marker_color='#e77200',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția creatininei',
                xaxis_title_text='Nivel creatinină',
                yaxis_title_text='Număr',
                bargap=0.05,
            )
            st.plotly_chart(fig)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x = df['serum_sodium'],
                xbins = dict(
                    start=113,
                    end=148,
                    size=1
                ),
                marker_color='#AAF0D1',
                opacity=1
            ))

            fig.update_layout(
                title_text='Distribuția nivelului de sodiu',
                xaxis_title_text='Sodiu',
                yaxis_title_text='Număr',
                bargap=0.05,
            )
            st.plotly_chart(fig)


        if option == "Distribuții comparative cu probabilitatea de moarte":
            st.header('Grafice ce ne ajută să observăm ce influență au atributele asupra morții')
            fig = px.histogram(df, x='age', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția age VS DEATH_EVENT", labels={"age": "AGE"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

            fig = px.histogram(df, x='creatinine_phosphokinase', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția creatinine_phosphokinase VS DEATH_EVENT", labels={"creatinine_phosphokinase": "creatinine_phosphokinase"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

            fig = px.histogram(df, x='ejection_fraction', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția ejection_fraction VS DEATH_EVENT", labels={"ejection_fraction": "ejection_fraction"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

            fig = px.histogram(df, x='platelets', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția platelets VS DEATH_EVENT", labels={"platelets": "platelets"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

            fig = px.histogram(df, x='serum_creatinine', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția serum_creatinine VS DEATH_EVENT", labels={"serum_creatinine": "serum_creatinine"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

            fig = px.histogram(df, x='serum_sodium', color="DEATH_EVENT", marginal='violin', hover_data=df.columns,
                               title="Distribuția serum_sodium VS DEATH_EVENT", labels={"serum_sodium": "serum_sodium"},
                               color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                               )
            st.plotly_chart(fig)

        if option == 'Diagrame de tip "pie"':
            d1 = df[(df["DEATH_EVENT"] == 0) & (df["sex"] == 1)]
            d2 = df[(df["DEATH_EVENT"] == 1) & (df["sex"] == 1)]
            d3 = df[(df["DEATH_EVENT"] == 0) & (df["sex"] == 0)]
            d4 = df[(df["DEATH_EVENT"] == 1) & (df["sex"] == 0)]
            label1 = ["Bărbat", "Femeie"]
            label2 = ['Bărbat - Surviețuitori', 'Bărbat - Morți', "Femeie -  Surpraviețuitoare", "Femei - Moarte"]
            values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
            values2 = [len(d1), len(d2), len(d3), len(d4)]
            # Create subplots: use 'domain' type for Pie subplot
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=label1, values=values1, name="GEN"),
                          1, 1)
            fig.add_trace(go.Pie(labels=label2, values=values2, name="GEN VS DEATH_EVENT"),
                          1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent")
            fig.update_layout(
                title_text="Distribuția genului în setul de date și GENDER VS DEATH_EVENT",
                # Add annotations in the center of the donut pies.
                annotations=[dict(text='GEN', x=0.21, y=0.5, font_size=9, showarrow=False),
                             dict(text='GEN VS DEATH', x=0.82, y=0.5, font_size=9, showarrow=False)],
                autosize=False, width=1000, height=400, paper_bgcolor="white")
            st.plotly_chart(fig)

            d1 = df[(df["DEATH_EVENT"] == 0) & (df["diabetes"] == 1)]
            d2 = df[(df["DEATH_EVENT"] == 1) & (df["diabetes"] == 1)]
            d3 = df[(df["DEATH_EVENT"] == 0) & (df["diabetes"] == 0)]
            d4 = df[(df["DEATH_EVENT"] == 1) & (df["diabetes"] == 0)]
            label1 = ["Bărbat", "Femeie"]
            label2 = ['Bărbat - Surviețuitori', 'Bărbat - Morți', "Femeie -  Surpraviețuitoare", "Femei - Moarte"]
            values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
            values2 = [len(d1), len(d2), len(d3), len(d4)]
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=label1, values=values1, name="DIABET"),
                          1, 1)
            fig.add_trace(go.Pie(labels=label2, values=values2, name="DIABET VS DEATH"),
                          1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent")
            fig.update_layout(
                title_text="Distribuția persoanelor cu diabet în setul de date și DIABET VS DEATH",
                # Add annotations in the center of the donut pies.
                annotations=[dict(text='DIABET', x=0.20, y=0.5, font_size=9, showarrow=False),
                             dict(text='DIABET VS DEATH', x=0.84, y=0.5, font_size=9, showarrow=False)],
                autosize=False, width=1000, height=400, paper_bgcolor="white")

            st.plotly_chart(fig)
            d1 = df[(df["DEATH_EVENT"] == 0) & (df["anaemia"] == 1)]
            d2 = df[(df["DEATH_EVENT"] == 1) & (df["anaemia"] == 1)]
            d3 = df[(df["DEATH_EVENT"] == 0) & (df["anaemia"] == 0)]
            d4 = df[(df["DEATH_EVENT"] == 1) & (df["anaemia"] == 0)]
            label1 = ["Bărbat", "Femeie"]
            label2 = ['Bărbat - Surviețuitori', 'Bărbat - Morți', "Femeie -  Surpraviețuitoare", "Femei - Moarte"]
            values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
            values2 = [len(d1), len(d2), len(d3), len(d4)]
            # Create subplots: use 'domain' type for Pie subplot
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=label1, values=values1, name="ANEMIA"),
                          1, 1)
            fig.add_trace(go.Pie(labels=label2, values=values2, name="ANEMIA VS DEATH"),
                          1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent")
            fig.update_layout(
                title_text="Distribuția persoanelor cu anemie în setul de date și ANEMIA VS DEATH",
                annotations=[dict(text='ANEMIE', x=0.20, y=0.5, font_size=9, showarrow=False),
                             dict(text='ANEMIE VS DEATH', x=0.84, y=0.5, font_size=9, showarrow=False)],
                autosize=False, width=1000, height=400, paper_bgcolor="white")
            st.plotly_chart(fig)

            d1 = df[(df["DEATH_EVENT"] == 0) & (df["high_blood_pressure"] == 1)]
            d2 = df[(df["DEATH_EVENT"] == 1) & (df["high_blood_pressure"] == 1)]
            d3 = df[(df["DEATH_EVENT"] == 0) & (df["high_blood_pressure"] == 0)]
            d4 = df[(df["DEATH_EVENT"] == 1) & (df["high_blood_pressure"] == 0)]
            label1 = ["Bărbat", "Femeie"]
            label2 = ['Bărbat - Surviețuitori', 'Bărbat - Morți', "Femeie -  Surpraviețuitoare", "Femei - Moarte"]
            values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
            values2 = [len(d1), len(d2), len(d3), len(d4)]
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=label1, values=values1, name="BP"),
                          1, 1)
            fig.add_trace(go.Pie(labels=label2, values=values2, name="BP VS DEATH"),
                          1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent")
            fig.update_layout(
                title_text="Distribuția persoanelor cu hipertensiune în setul de date și BP VS DEATH",
                annotations=[dict(text='BP', x=0.20, y=0.5, font_size=9, showarrow=False),
                             dict(text='BP VS DEATH', x=0.84, y=0.5, font_size=9, showarrow=False)],
                autosize=False, width=1000, height=400, paper_bgcolor="white")
            st.plotly_chart(fig)

            d1 = df[(df["DEATH_EVENT"] == 0) & (df["smoking"] == 1)]
            d2 = df[(df["DEATH_EVENT"] == 1) & (df["smoking"] == 1)]
            d3 = df[(df["DEATH_EVENT"] == 0) & (df["smoking"] == 0)]
            d4 = df[(df["DEATH_EVENT"] == 1) & (df["smoking"] == 0)]
            label1 = ["Bărbat", "Femeie"]
            label2 = ['Bărbat - Surviețuitori', 'Bărbat - Morți', "Femeie -  Surpraviețuitoare", "Femei - Moarte"]
            values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
            values2 = [len(d1), len(d2), len(d3), len(d4)]
            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=label1, values=values1, name="SMOKING"),
                          1, 1)
            fig.add_trace(go.Pie(labels=label2, values=values2, name="SMOKING VS DEATH"),
                          1, 2)
            fig.update_traces(hole=.4, hoverinfo="label+percent")

            fig.update_layout(
                title_text="Distribuția persoanelor fumătoare în setul de date și SMOKING VS DEATH",
                # Add annotations in the center of the donut pies.
                annotations=[dict(text='SMOKING', x=0.20, y=0.5, font_size=9, showarrow=False),
                             dict(text='SMOKING VS DEATH', x=0.84, y=0.5, font_size=9, showarrow=False)],
                autosize=False, width=1000, height=400, paper_bgcolor="white")
            st.plotly_chart(fig)

    elif choice == 'Diagnosticare':
        st.title('Diagnosticarea insuficienței cardiace')
        st.header('Introduceți datele rezultate din analizele pacientului')

        age = st.number_input("Vârstă", int(df['age'].min()), int(df['age'].max()))
        anaemia = st.radio("Pacientul suferă de anemie?", tuple(feature_dict.keys()))
        creatinine_phosphokinase = st.number_input("Enzime CPK (mcg/L)", int(df['creatinine_phosphokinase'].min()), int(df['creatinine_phosphokinase'].max()))
        diabetes = st.radio("Pacientul suferă de diabet?", tuple(feature_dict.keys()))
        ejection_fraction = st.number_input("Procentul de ieșire a sângelui (%)", int(df['ejection_fraction'].min()), int(df['ejection_fraction'].max()))
        high_blood_pressure = st.radio("Pacientul suferă de hipertensiune?", tuple(feature_dict.keys()))
        platelets = st.number_input("Trombocite (kilotrombocite/mL)", df['platelets'].min(), df['platelets'].max())
        serum_creatinine = st.number_input("Nivelul de creatinină (mg/dL)", df['serum_creatinine'].min(), df['serum_creatinine'].max())
        serum_sodium = st.number_input("Nivelul de sodiu (mEq/L)", 114, 148)
        sex = st.radio("Sex",tuple(gender_dict.keys()))
        smoking = st.radio("Pacientul este fumător?", tuple(feature_dict.keys()))
        time = st.number_input("Perioada de urmărire (zile)", int(df['time'].min()), int(df['time'].max()))

        feature_list = [age,get_feature_value(anaemia),creatinine_phosphokinase,get_feature_value(diabetes),ejection_fraction,get_feature_value(high_blood_pressure),platelets,serum_creatinine,serum_sodium,get_value(sex, gender_dict),get_feature_value(smoking),time]

        if st.checkbox("Datele introduse:"):
            st.write(feature_list)
            pretty_result = {"age": age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase, 'diabetes': diabetes, 'ejection_fraction': ejection_fraction, \
                             'high_blood_pressure': high_blood_pressure, 'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium, \
                             'sex': sex, 'smoking': smoking, 'time': time}

            st.json(pretty_result)

        # feature_list = [ejection_fraction,serum_creatinine,time]

        single_sample = np.array(feature_list).reshape(1, -1)
        # sc = StandardScaler()
        # single_sample = sc.fit_transform(single_sample)

        model_choice = st.selectbox("Selectează modelul ML: ", ["Logistic Regression", "K Nearest Neighbour", "Decision Tree", "RandomForest"])
        if st.button("Afișează rezultatele"):
            if model_choice == "Logistic Regression":
                loaded_model = load_model("models/logistic_regression_model.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)

            elif model_choice == "K Nearest Neighbour":
                loaded_model = load_model("models/k_neighbours_classifier.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)

            elif model_choice == "Decision Tree":
                loaded_model = load_model("models/decision_tree_classifier.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)

            elif model_choice == "RandomForest":
                loaded_model = load_model("models/randomforest_classifier.pkl")
                prediction = loaded_model.predict(single_sample)
                pred_prob = loaded_model.predict_proba(single_sample)

            st.write(prediction)


            if prediction == 0:
                st.success("Patcient ce nu are riscuri să moară din cauza insuficienței cardiace.")
            else:
                st.warning("Patcient ce are mari riscuri să moară din cauza insuficienței cardiace.")

            pred_probability_score = {"Fără riscuri": pred_prob[0][0], "Cu riscuri": pred_prob[0][1]}

            st.subheader("Probabilitatea de predicție pentru algoritmul {} este:".format(model_choice))
            st.json(pred_probability_score)

    elif choice == 'Feedback':
        st.title('Feedback')
        st.header("Vă rog răspundeți la următoarele întrebări pentru îmbunătățirea aplicației:")
        interaction = st.slider("Cât de bine ați putut interacționa cu aplicația?", min_value=1, max_value=5)
        results = st.slider("Cât de corecte vi se par rezultatele aplicației?", min_value=1, max_value=5)
        speed = st.slider("Cât de rapidă vi s-a părut aplicația?", min_value=1, max_value=5)
        suggestions = st.text_area("Sugestii/ observații: ")

        if st.button("Trimiteți feedback-ul"):
            create_feedbacktable()
            add_feedbackdata(interaction, results, speed, suggestions)
            st.success("Feedback trimis cu succes! Vă mulțumim!")

if __name__ == '__main__':
    main()