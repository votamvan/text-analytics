import logging
import json
import falcon
from langdetect import detect
import numpy as np
from sentiment.config import RESOURCE_PATH

# from sentiment.ai.predict_with_classifier import fastai_load_model, predict_text
# from sentiment.train import SimpleModel
from sentiment.ai.models import ggbert

logger = logging.getLogger()

class Demo(object):
    def on_get(self, req, resp, **kwargs):
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        html_path = RESOURCE_PATH/"html/text-analytics-api.html"
        with html_path.open(mode='r') as f:
            resp.body = f.read()

    def on_post(self, req, resp, **kwargs):
        try:
            api_return = do_post(req)
            print(api_return)
            resp.media = api_return
        except Exception as err:
            logger.error(err, exc_info=True)
            resp.media = {"err": "{}".format(err)}

lang_map = {
    'vi': 'Vietnamese',
    'en': 'English',
    'th': 'Thai',
    'ja': 'Japanese'
}
def do_post(req):
    print(req.content_type)
    if 'application/json' not in req.content_type:
        text = req.get_param('Query')
    _lang = lang_map.get(detect(text), "unknown")
    prob = ggbert.predict(text)
    print("prob=", prob, text)
    return {
        "Query": text,
        "lang": _lang,
        # "pred": int(100*prob)
        "pred": int(100*(1-prob))
    }
