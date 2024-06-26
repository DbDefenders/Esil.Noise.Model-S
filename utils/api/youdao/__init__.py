# -*- coding: utf-8 -*-
# Author: Vi
# Created on: 2024-06-25 14:29:37
# Description: 有道api

import os
from ..common import doCall
from ..common import addAuthParams

def translate_text(q, lang_from='zh-CHS',lang_to='en',vocab_id = None, api_id=None, api_key=None):
    if not api_id:
        api_id = os.getenv("YOUDAO_API_ID")
        
    if not api_key: 
        api_key = os.getenv("YOUDAO_API_KEY")
        
    if not api_id or not api_key:
        raise ValueError("Youdao API_ID or API_KEY is not provided.")
    
    data = {'q': q, 'from': lang_from, 'to': lang_to, 'vocabId': vocab_id}

    addAuthParams(api_id, api_key, data)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    res = doCall('https://openapi.youdao.com/api', header, data, 'post')
    
    if res.status_code == 200:
        js = res.json()
        translation = js.get('translation')
        return translation    
