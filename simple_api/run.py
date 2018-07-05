#!flask/bin/python
from flask import Flask, jsonify, session
from simple_api.modules.tickets import Predictor as Predictor


app = Flask(__name__)

'''
@app.route('/predict/rg/<ir_no>', methods=['GET'])
def index(ir_no):
    px = Prediction(ir_no)
    ret = px.resolve("assignee_grp")
    return ret[0]

@app.route('/predict/ir/<ir_no>', methods=['GET'])
def index(ir_no):
    px = Prediction(ir_no)
    ret = px.resolve("assignee_grp")
    return ret[0]
'''

@app.route('/simpel_api/auth', methods=['GET'])
def authenticate():
    a = Authentication()
    ret = a.authenticate()
    return ret

# Print the site map on the console and browser
@app.route("/site-map")
def site_map():
    links = []
    func_list = {}
    #import pdb; pdb.set_trace()
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static':
            func_list[rule.rule] = app.view_functions[rule.endpoint].__dict__

        print(rule.__str__)
        links.append(rule.__str__)
    return jsonify(func_list)
    #TODO: On the browser the options are not displayed
    return '</br>'.join(map(str, links))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
