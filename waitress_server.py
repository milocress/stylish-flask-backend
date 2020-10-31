from waitress import serve
import poc
serve(poc.app, host='https://stylish-videos.herokuapp.com/', port=80)
