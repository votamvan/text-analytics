import logging
import falcon
from sentiment.web import demo
from sentiment.config import RESOURCE_PATH

log_format = '%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger()
handler = logging.FileHandler(RESOURCE_PATH/"logs/console.log")
handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(handler)

api = falcon.API()
api.add_route('/demo', demo.Demo())
api.add_static_route('/static', str(RESOURCE_PATH/'html'))
api.req_options.auto_parse_form_urlencoded = True
logger.info("Text Analytics API server started")

if __name__ == '__main__':
    from wsgiref import simple_server
    httpd = simple_server.make_server('127.0.0.1', 8000, api)
    httpd.serve_forever()