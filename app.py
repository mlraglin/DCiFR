from flask import Flask # importing the flask class
app = Flask(__name__) # creating an instance of the Flask class

@app.route('/') # The primary url for our application
def hello_world(): # This method returns 'Flask Dockerized', which is displayed in our browser.
    execfile(dcifr.py)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

