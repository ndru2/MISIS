"""Streamlit-клиент. Запуск: streamlit run src/pdfscan/web/app.py."""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API_URL = os.environ.get('RAG_API_URL', 'http://127.0.0.1:8000')


def _pages(pages):
    return ', '.join(str(page) for page in pages) if pages else 'не указаны'


def show_evidence(items, cited_ids):
    with st.expander(f'Источники RAG ({len(items)})', expanded=False):
        for index, item in enumerate(items, 1):
            used = item.get('unit_id') in cited_ids
            title = (f'[E{index}] {item.get("unit_type", "chunk")} · '
                     f'{item.get("document_id") or "без документа"} · '
                     f'стр. {_pages(item.get("pages", []))}')
            st.markdown(f'**{title}**')
            st.caption(('Использован в ответе. ' if used else 'Не использован в ответе. ') +
                       'Каналы: ' + ', '.join(item.get('retrieval_channels', [])) +
                       f'; RRF: {item.get("score", 0):.4f}; '
                       f'relevance: {item.get("relevance_score", 0):.4f}')
            if item.get('structured'):
                st.json(item['structured'], expanded=False)
            st.text(item.get('text') or item.get('context') or '')
            if index < len(items):
                st.divider()


def show_answer(payload):
    st.markdown(payload['answer'])
    citations = payload.get('citations') or []
    mode = payload.get('generation_mode', 'llm_only')
    st.caption('Режим: ' + ('LLM + подтверждённый RAG' if mode == 'hybrid'
                             else 'только LLM: релевантных Evidence не найдено'))
    if mode == 'hybrid':
        with st.expander('Самостоятельный черновик LLM', expanded=False):
            st.markdown(payload.get('base_answer') or '')
    if citations:
        st.caption('Цитируемые объекты: ' + ', '.join(citations))
    profile = payload.get('profile') or {}
    with st.expander('Как был найден контекст', expanded=False):
        st.write('Векторные каналы:', ', '.join(profile.get('vector_kinds', [])))
        st.write('Итоговые каналы:', ', '.join(profile.get('channels', [])))
        st.write('RRF-кандидатов:', profile.get('candidate_count', 0),
                 '; после reranker:', profile.get('accepted_count', 0),
                 '; порог:', profile.get('relevance_threshold', 0))
        if profile.get('filters'):
            st.json(profile['filters'], expanded=False)
    show_evidence(payload.get('evidence') or [], set(citations))


st.set_page_config(page_title='Metallurgy RAG', page_icon='⚗️', layout='wide')
st.title('Metallurgy RAG')
st.caption('Ответы Qwen с проверяемыми фрагментами корпуса: текст, формулы, таблицы и единицы.')

with st.sidebar:
    st.header('Подключение')
    api_url = st.text_input('FastAPI URL', value=DEFAULT_API_URL).rstrip('/')
    provider = st.selectbox('LLM provider', ('ollama', 'openai-compatible'))
    model = st.text_input('Модель', value=os.environ.get('RAG_LLM_MODEL', 'qwen3:8b'))
    base_url = st.text_input('LLM URL (необязательно)', value='')
    evidence_k = st.slider('Число Evidence', min_value=1, max_value=12, value=6)
    if st.button('Очистить диалог', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption('Qdrant и retrieval настраиваются на FastAPI-сервере, а не в браузере.')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for item in st.session_state.messages:
    with st.chat_message(item['role']):
        if item['role'] == 'assistant':
            show_answer(item['payload'])
        else:
            st.markdown(item['content'])

question = st.chat_input('Например: как температура влияет на восстановление Fe3O4?')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Ищу evidence и формирую ответ…'):
            request = {
                'question': question,
                'evidence_k': evidence_k,
                'provider': provider,
                'model': model,
            }
            if base_url.strip():
                request['base_url'] = base_url.strip()
            try:
                response = httpx.post(f'{api_url}/api/answer', json=request, timeout=240)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                st.error(f'Не удалось получить ответ от FastAPI: {exc}')
                payload = None
        if payload:
            show_answer(payload)
            st.session_state.messages.append({'role': 'assistant', 'payload': payload})
