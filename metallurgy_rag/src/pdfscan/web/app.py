"""Streamlit-клиент. Запуск: streamlit run src/pdfscan/web/app.py."""

from __future__ import annotations

import os

import httpx
import streamlit as st

from pdfscan.env import load_project_env

load_project_env()

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


_MODE_LABELS = {
    'hybrid': '✅ RAG-ответ подтверждён LLM Judge',
    'hybrid_partial': '⚠️ RAG-ответ, но Judge не достиг порога покрытия',
    'llm_only': '🤖 только LLM (явный opt-in, без подтверждения источниками)',
    'no_evidence': '🚫 релевантных документов не найдено',
}


def show_answer(payload, *, on_fallback=None, key=''):
    profile = payload.get('profile') or {}
    st.caption('Методы поиска: ' + ', '.join(profile.get('methods', [])))
    mode = payload.get('generation_mode', 'no_evidence')
    if mode == 'no_evidence':
        st.warning(payload['answer'])
        if on_fallback is not None:
            if st.button('Сгенерировать ответ силами LLM (без подтверждения источниками)',
                         key=f'fallback-{key}'):
                on_fallback()
        return
    st.markdown(payload['answer'])
    citations = payload.get('citations') or []
    st.caption('Режим: ' + _MODE_LABELS.get(mode, mode))
    if mode in ('hybrid', 'hybrid_partial'):
        st.caption(
            f'Judge покрытие: {payload.get("judge_score", 0):.2f} · '
            f'модель judge: {payload.get("judge_model") or "—"} · '
            f'итераций: {payload.get("iterations", 0)} · '
            f'фрагментов использовано: {payload.get("evidence_used", 0)}')
        if payload.get('judge_missing'):
            st.caption('Judge отметил как не покрытое: ' + payload['judge_missing'])
    if citations:
        st.caption('Цитируемые объекты: ' + ', '.join(citations))
    profile = payload.get('profile') or {}
    with st.expander('Как был найден контекст', expanded=False):
        st.write('Методы:', ', '.join(profile.get('methods', [])))
        st.write('Кандидаты по каналам:', profile.get('channel_counts', {}))
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
st.caption('Ответы по проверяемым фрагментам корпуса: текст, формулы, таблицы и единицы.')

with st.sidebar:
    st.header('Подключение')
    api_url = st.text_input('FastAPI URL', value=DEFAULT_API_URL).rstrip('/')
    providers = ('ollama', 'openai-compatible')
    default_provider = os.environ.get('RAG_LLM_PROVIDER', 'ollama')
    def reset_generator():
        if st.session_state.llm_provider == 'ollama':
            st.session_state.generator_model = 'qwen3:8b'
            st.session_state.generator_url = os.environ.get(
                'RAG_OLLAMA_URL', 'http://localhost:11434')
        else:
            st.session_state.generator_model = 'nn-tech/MetalGPT-1:featherless-ai'
            st.session_state.generator_url = 'https://router.huggingface.co/v1'

    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = (
            default_provider if default_provider in providers else 'ollama')
        reset_generator()
        st.session_state.generator_model = os.environ.get(
            'RAG_LLM_MODEL') or st.session_state.generator_model
        st.session_state.generator_url = os.environ.get(
            'RAG_LLM_BASE_URL') or st.session_state.generator_url
    provider = st.selectbox(
        'LLM provider', providers, key='llm_provider', on_change=reset_generator,
        help='Ollama — локальный Qwen; OpenAI-compatible — MetalGPT через Hugging Face.')
    model = st.text_input(
        'Модель', key='generator_model',
        help='При смене провайдера подставляется соответствующая модель.')
    base_url = st.text_input(
        'LLM URL', key='generator_url',
        help='Адрес автоматически меняется вместе с провайдером; его можно изменить вручную.')
    judge_model = st.text_input(
        'LLM Judge model (необязательно)', value='',
        help=('Пусто — использовать Judge из конфигурации API. Заполните model slug, '
              'например qwen/qwen3.6-35b-a3b; ключ OpenRouter остаётся на сервере.'))
    methods = st.multiselect(
        'Методы поиска (любое сочетание)', ['vector', 'bm25', 'graph'],
        default=['vector', 'bm25'],
        format_func=lambda name: {'vector': 'Вектор', 'bm25': 'Индекс (BM25)',
                                  'graph': 'Граф (Neo4j)'}[name])
    if not methods:
        st.warning('Выберите хотя бы один метод поиска.')
    evidence_k = st.slider('Число фрагментов источников', min_value=1, max_value=12, value=6)
    if st.button('Очистить диалог', use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption('Qdrant и retrieval настраиваются на FastAPI-сервере, а не в браузере.')

def ask_api(question: str, *, allow_llm_fallback: bool = False,
            selected_methods=None) -> dict:
    request = {
        'question': question,
        'methods': methods if selected_methods is None else selected_methods,
        'evidence_k': evidence_k,
        'provider': provider,
        'model': model,
        'allow_llm_fallback': allow_llm_fallback,
    }
    if judge_model.strip():
        request['judge_model'] = judge_model.strip()
    if base_url.strip():
        request['base_url'] = base_url.strip()
    response = httpx.post(f'{api_url}/api/answer', json=request, timeout=240)
    if response.is_error:
        try:
            detail = response.json().get('detail')
        except ValueError:
            detail = None
        if detail:
            raise RuntimeError(f'HTTP {response.status_code}: {detail}')
    response.raise_for_status()
    return response.json()


if 'messages' not in st.session_state:
    st.session_state.messages = []

for index, item in enumerate(st.session_state.messages):
    with st.chat_message(item['role']):
        if item['role'] == 'assistant':
            def _fallback(index=index, question=item.get('question', ''),
                          selected=item['payload'].get('profile', {}).get('methods')):
                try:
                    st.session_state.messages[index]['payload'] = ask_api(
                        question, allow_llm_fallback=True, selected_methods=selected)
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(f'Не удалось получить ответ от FastAPI: {exc}')
                    return
                st.rerun()
            show_answer(item['payload'], on_fallback=_fallback, key=str(index))
        else:
            st.markdown(item['content'])

question = st.chat_input('Например: как температура влияет на восстановление Fe3O4?',
                         disabled=not methods)
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Ищу фрагменты источников и формирую ответ…'):
            try:
                payload = ask_api(question)
            except (httpx.HTTPError, RuntimeError) as exc:
                st.error(f'Не удалось получить ответ от FastAPI: {exc}')
                payload = None
        if payload:
            new_index = len(st.session_state.messages)

            def _fallback(index=new_index, question=question, selected=tuple(methods)):
                try:
                    st.session_state.messages[index]['payload'] = ask_api(
                        question, allow_llm_fallback=True, selected_methods=selected)
                except (httpx.HTTPError, RuntimeError) as exc:
                    st.error(f'Не удалось получить ответ от FastAPI: {exc}')
                    return
                st.rerun()
            show_answer(payload, on_fallback=_fallback, key=str(new_index))
            st.session_state.messages.append(
                {'role': 'assistant', 'payload': payload, 'question': question})
