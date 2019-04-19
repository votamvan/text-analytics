#!/bin/bash

gunicorn sentiment.app:api --reload

