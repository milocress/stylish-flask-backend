from waitress import serve

import upload

serve(upload.app, host="0.0.0.0", port=80)
