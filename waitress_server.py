from waitress import serve
import poc
serve(poc.app, host='0.0.0.0', port=80)
