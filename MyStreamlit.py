import streamlit as st

st.image('./pic/Breast - Copy.webp')
col1, col2 = st.columns(2)
with col1:
  st.header('ธัญสิทธิ์ เดชสิทธิ์ธาวิน')
with col2:
  st.subheader('สาขาวิชาวิทยาการข้อมูล')
  st.text('คณะวิทยาศาสตร์และเทคโนโลยี')

  html_1 = """
<div style="background-color:#76D7C4;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>การทำนายข้อมูลดอกไม้เบื้องต้น</h4></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

import pandas as pd
dt=pd.read_csv('./data/iris.csv')
st.write(dt.head(10))

dt1 = dt['ClumpThickness'].mean()
dt2 = dt['UniformityofCellSize'].mean()
dt3 = dt['UniformityofCellShape'].mean()
dt4 = dt['MarginalAdhesion'].mean()
dt5 = dt['SingleEpithelialCellSize'].mean()
dt6 = dt['BareNuclei'].mean()
dt7 = dt['BlandChromatin'].mean()
dt8 = dt['NormalNucleoli'].mean()
dt9 = dt['Mitoses'].mean()

dx = [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9]
dx2 = pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])
if st.button("show bar chart"):
    st.bar_chart(dx2)
    st.button("Not show bar chart")
else :
    st.button("Not show bar chart") 

html_2 = """
<div style="background-color:#FFBF00;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายการเป็นโรคมะเร็ง</h5></center>
</div>
"""
st.markdown(html_2, unsafe_allow_html=True)
st.markdown("")   

input1 = st.number_input("กรุณาเลือกข้อมูล input1")
input2 = st.number_input("กรุณาเลือกข้อมูล input2")
input3 = st.number_input("กรุณาเลือกข้อมูล input3")
input4 = st.number_input("กรุณาเลือกข้อมูล input4")
input5 = st.number_input("กรุณาเลือกข้อมูล input5")
input6 = st.number_input("กรุณาเลือกข้อมูล input6")
input7 = st.number_input("กรุณาเลือกข้อมูล input7")
input8 = st.number_input("กรุณาเลือกข้อมูล input8")
input9 = st.number_input("กรุณาเลือกข้อมูล input9")



from sklearn.neighbors import KNeighborsClassifier
import numpy as np

if st.button("ทำนายผล"):
   # ทำนาย
   #dt = pd.read_csv("./data/iris.csv") 

   X = dt.drop('variety', axis=1)
   y = dt.variety   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

    #ข้อมูล input สำหรับทดลองจำแนกข้อมูล
   x_input = np.array([[input1,input2,input3,input4,input5,input6,input7,input8,input9]])
    # เอา input ไปทดสอบ
   st.write(Knn_model.predict(x_input))
   out=Knn_model.predict(x_input)

   if out[0]=="2":
      st.image("./pic/iris1.jpg")
      st.header("ไม่เป็นมะเร็ง")
   elif out[0]=="เป็นมะเร็ง":
      st.image("./pic/iris2.jpg")
      st.header("Versicolor")
   else:
      st.image("./pic/iris3.jpg")  
      st.header("Verginiga")
   st.button("ไม่ทำนายผล")
else :
    st.button("ไม่ทำนายผล")

