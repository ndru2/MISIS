"""One short real request per configured model, without printing credentials."""

import sys

import httpx

from pdfscan.web.api import ApiSettings
from pdfscan.rag.llm import create_llm


def main():
    settings = ApiSettings.from_env()
    clients = {
        'Generator': create_llm(provider=settings.provider, model=settings.model,
                                base_url=settings.llm_base_url, timeout_seconds=90),
        'Judge': create_llm(provider=settings.judge_provider or settings.provider,
                            model=settings.judge_model or settings.model,
                            base_url=settings.judge_base_url or settings.llm_base_url,
                            api_key=settings.judge_api_key, timeout_seconds=90),
    }
    failed = False
    for role, client in clients.items():
        print(f'{role}: {client.model}', flush=True)
        try:
            text = client.generate([
                {'role': 'system', 'content': 'Отвечай кратко, одним предложением. /no_think'},
                {'role': 'user', 'content': 'Что такое штейн в металлургии? /no_think'},
            ])
            print(f'  OK: получен непустой ответ ({len(text)} символов)', flush=True)
        except Exception as exc:
            failed = True
            cause = exc.__cause__
            if isinstance(cause, httpx.HTTPStatusError):
                print(f'  FAIL: HTTP {cause.response.status_code}', flush=True)
            else:
                print(f'  FAIL: {type(exc).__name__}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
