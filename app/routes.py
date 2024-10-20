from flask import Blueprint, render_template
from . import db  # Import db, not app

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('index.html')
