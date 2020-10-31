from waitress import serve
import upload
serve(upload.app, host='stylish-videos.herokuapp.com', port=80)
