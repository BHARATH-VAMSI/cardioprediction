from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping

data = pd.read_csv("cardiovasculardataset.csv", delimiter=";")
data.drop("id", axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/submit', methods = ["POST"])
def submit():
    age = request.form.get('age')
    gender = request.form.get('gender')
    height = request.form.get('height')
    weight = request.form.get('weight')
    ap_hi = request.form.get('ap_hi')
    ap_lo = request.form.get('ap_lo')
    cholestrol = request.form.get('cholestrol')
    glucose = request.form.get('glucose')
    smoke = request.form.get('smoke')
    alcohol = request.form.get('alcohol')
    active = request.form.get('active')
    user_input = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholestrol, glucose, smoke, alcohol, active]])
    user_input = sc.transform(user_input)
    user_input = np.reshape(user_input, (user_input.shape[0], 1, user_input.shape[1]))
    prediction = model.predict(user_input)

    return render_template('index1.html', res=prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)