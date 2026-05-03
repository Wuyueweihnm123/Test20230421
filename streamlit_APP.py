# 导入需要的库
import streamlit as st
import pandas as pd
import joblib


st.header('Predicting the probability of hypertension in the general population using machine learning')
# 输入框
st.sidebar.header('Variables')
b = st.sidebar.selectbox("Age, years", ("18～30", "31～40",
                                    "41～50", "51～60", "＞60"))
c = st.sidebar.selectbox("Smoking", ("No", "Yes"))
d = st.sidebar.selectbox("High UALB，UALB＞20μg/ml", ("No", "Yes"))
e = st.sidebar.selectbox("Educational level",
                     ("Not educated", "Primary school",
                      "Middle school", "High school", "College or above"))
f = st.sidebar.selectbox("DM", ("No", "Yes"))
g = st.sidebar.selectbox("BMI, kg/m2",
                     ("≤18.4，Underweight", "18.5～23.9，Normal",
                      "24～27.9，Overweight", "≥28，Obesity"))
h = st.sidebar.selectbox("Sex", ("Female", "Male"))
i = st.sidebar.selectbox("High TG，TG≥2.26mmol/L", ("No", "Yes"))
j = st.sidebar.selectbox("Income level，thousand/year", ("＜10", "10～20",
                                                    "21～30", "31～50",
                                                    "51～100", "＞100"))
k = st.sidebar.selectbox("Family history of hypertension", ("No", "Yes"))
# 如果按下按钮
if st.button("Predict"):  # 显示按钮
    # 加载训练好的模型
    model_XGB = joblib.load("models/model_XGB.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[b,c,d,e,f,g,h,i,j,k]],
                     columns = ["Age, years", "Smoking", "High UALB，UALB＞20μg/ml",
                                "Educational level", "DM", "BMI, kg/m2",
                                "Sex", "High TG，TG≥2.26mmol/L", "Income level，thousand/year",
                                "Family history of hypertension"
                                ])

    X = X.replace(["18～30", "31～40","41～50", "51～60", "＞60"],
                                [1, 2, 3, 4, 5])
    X = X.replace(["No", "Yes"],
                                [0, 1])
    X = X.replace(["Not educated", "Primary school",
                          "Middle school", "High school", "College or above"],
                                [1, 2, 3, 4, 5])
    X = X.replace(["≤18.4，Underweight", "18.5～23.9，Normal",
                          "24～27.9，Overweight", "≥28，Obesity"],
                                [1, 2, 3, 4])
    X = X.replace(["Female", "Male"],
                                [0, 1])
    X = X.replace(["＜10", "10～20",
                "21～30", "31～50",
                "51～100", "＞100"],
                [1, 2, 3, 4, 5, 6])


    # 进行预测
    prediction = model_XGB.predict(X)[0]
    Predict_proba = model_XGB.predict_proba(X)[:, 1][0]
    # 输出预测结果
    if prediction == 0:
        st.subheader(f"Model predicted outcome for hypertension:  NO")
    else:
        st.subheader(f"Model predicted outcome for hypertension:  YES")
    st.subheader(f"Probability of predicting hypertension:  {'%.2f'%float(Predict_proba*100)+'%'}")

