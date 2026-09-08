"""Оценка качества раскладки формул на химию и математику.

Обучающий набор классификатора почти весь синтетический — формулы из учебников,
набранные чистым LaTeX. Корпус же приходит из MinerU, и его записи выглядят
иначе, поэтому macro F1 с обучения к раскладке отношения не имеет: измерять
качество нужно на самом корпусе.

Здесь считаются три разные вещи, и путать их нельзя.

*Уверенность модели* — то, что она сама думает о своём ответе. Это не точность:
модель бывает уверенно неправа, но по доле ответов у порога видно, какая часть
раскладки держится на угадывании.

*Согласие с разбором формулы* по символам элементов. Разбор точен, но
срабатывает лишь там, где все вещества раскладываются без остатка, поэтому он
годится как эталон для одной оси — «химия против остального» — и только на своём
подмножестве.

*Ручная разметка* стратифицированной выборки — единственный честный замер. Файл
для неё готовит ``sample``, считает ``score``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from pdfscan import paths
from pdfscan.prepare.split_corpus import is_chemical

FORMULA_CATEGORIES = ('formulas_chem', 'formulas_math')

# Ниже этого порога ответ модели держится на угадывании: при трёх классах
# случайный выбор даёт 0.33, и всё, что рядом, содержательной опоры не имеет.
LOW_CONFIDENCE = 0.5

SAMPLE_COLUMNS = ('id', 'predicted', 'decided_by', 'confidence', 'true_class',
                  'stratum', 'stratum_size', 'doc_id', 'page', 'latex', 'context')


def _load_formulas(split_dir: Path) -> list[dict]:
    records = []
    for category in FORMULA_CATEGORIES:
        path = split_dir / 'all' / f'{category}.jsonl'
        if not path.exists():
            raise SystemExit(f'нет {path}; сначала: python -m pdfscan.prepare.split_corpus')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                record['category'] = category
                records.append(record)
    return records


def _model_probabilities(records: list[dict], model_path=None) -> list[dict]:
    """Ответ модели на каждой формуле, независимо от правила."""
    from pdfscan.formulas.classifier import FormulaClassifier

    model = FormulaClassifier.load(model_path or paths.FORMULA_CLASSIFIER)
    if model is None:
        raise SystemExit(f'нет модели {model_path or paths.FORMULA_CLASSIFIER}')

    classes = list(model.label_encoder.classes_)
    matrix = model._build_feature_matrix(
        [record['latex'] for record in records],
        [record.get('context', '') for record in records],
        fit_char=False)
    probabilities = model.boosting.predict_proba(matrix)
    return [dict(zip(classes, row.tolist())) for row in probabilities]


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(share):
        return round(ordered[min(len(ordered) - 1, int(share * len(ordered)))], 3)

    return {'p05': at(0.05), 'p25': at(0.25), 'median': at(0.5),
            'p75': at(0.75), 'mean': round(sum(ordered) / len(ordered), 3)}


def _wilson(hits: int, total: int) -> list[float]:
    """Доверительный интервал доли; на выборке в сотню строк он широкий."""
    if not total:
        return [0.0, 0.0]
    z, share = 1.96, hits / total
    center = (share + z * z / (2 * total)) / (1 + z * z / total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    spread /= 1 + z * z / total
    return [round(max(0.0, center - spread), 3), round(min(1.0, center + spread), 3)]


def _probabilities_cached(records: list[dict], cache_path: Path, model_path=None) -> list[dict]:
    """Ответы модели с кэшем: прогон BERT по корпусу занимает минуты.

    Подпись считается по самим формулам в их порядке. Одного счётчика записей
    не хватило бы: после перекладки формул между каталогами их столько же, но
    идут они иначе, и старые вероятности встали бы не к тем строкам.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(record['latex'].encode('utf-8'))
        digest.update(b'\0')
    signature = digest.hexdigest()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if cached.get('signature') == signature:
            print(f'ответы модели из кэша {cache_path.name}')
            return cached['probabilities']

    print('считаю ответы модели (это займёт несколько минут)...')
    probabilities = _model_probabilities(records, model_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({'signature': signature, 'probabilities': probabilities}),
        encoding='utf-8')
    return probabilities


def analyse(split_dir: Path, model_path=None, cache_path=None) -> dict:
    """Автоматические метрики: уверенность модели и согласие с разбором формулы."""
    records = _load_formulas(split_dir)
    print(f'{len(records)} формул в {split_dir}')
    probabilities = _probabilities_cached(
        records, Path(cache_path or paths.REPORTS_DIR / 'split_audit_cache.json'), model_path)

    by_rule = Counter()
    agreement = Counter()
    confidence_by_class = defaultdict(list)
    low_confidence = Counter()
    disagreement_examples = defaultdict(list)

    for record, proba in zip(records, probabilities):
        model_class = max(proba, key=proba.get)
        confidence = proba[model_class]
        rule_says_chemistry = is_chemical(record['latex'])

        record['model_class'] = model_class
        record['confidence'] = confidence
        record['decided_by'] = 'rule' if rule_says_chemistry else 'model'

        by_rule['rule' if rule_says_chemistry else 'model'] += 1
        confidence_by_class[model_class].append(confidence)
        if confidence < LOW_CONFIDENCE:
            low_confidence[model_class] += 1

        if rule_says_chemistry:
            agreement['rule_total'] += 1
            if model_class == 'chemistry':
                agreement['model_agrees'] += 1
            else:
                agreement[f'model_says_{model_class}'] += 1
                if len(disagreement_examples['rule_chemistry']) < 15:
                    disagreement_examples['rule_chemistry'].append({
                        'latex': record['latex'][:160],
                        'model_class': model_class,
                        'confidence': round(confidence, 3),
                    })
        elif model_class == 'chemistry' and len(disagreement_examples['model_only']) < 15:
            disagreement_examples['model_only'].append({
                'latex': record['latex'][:160],
                'confidence': round(confidence, 3),
            })

    total = len(records)
    recall = agreement['model_agrees'] / max(agreement['rule_total'], 1)
    return {
        'formulas': total,
        'decided_by': dict(by_rule),
        'final_categories': dict(Counter(record['category'] for record in records)),
        'model_classes': dict(Counter(record['model_class'] for record in records)),
        'confidence': {name: _quantiles(values)
                       for name, values in sorted(confidence_by_class.items())},
        'low_confidence': {
            'threshold': LOW_CONFIDENCE,
            'total': sum(low_confidence.values()),
            'share': round(sum(low_confidence.values()) / max(total, 1), 3),
            'by_class': dict(low_confidence),
        },
        # Разбор формулы точен, но узок: он эталон только на своём подмножестве,
        # и «recall» здесь — доля явной химии, которую модель узнала бы сама.
        'rule_as_reference': {
            'rule_positive': agreement['rule_total'],
            'model_agrees': agreement['model_agrees'],
            'model_recall_on_clear_chemistry': round(recall, 3),
            'model_recall_ci95': _wilson(agreement['model_agrees'], agreement['rule_total']),
            'model_says_math': agreement['model_says_math'],
            'model_says_physics': agreement['model_says_physics'],
        },
        'examples': dict(disagreement_examples),
        '_records': records,
    }


def build_sample(records: list[dict], size: int, seed: int) -> list[dict]:
    """Стратифицированная выборка: по классам и по уверенности модели.

    Ровная случайная выборка почти вся состоит из уверенной химии — её в корпусе
    больше половины, — и по ней не видно, где раскладка ошибается. Спорные
    записи набираются наравне с уверенными, поэтому доли в выборке не повторяют
    доли в корпусе, и общую точность по ней считать нельзя: ``score`` считает её
    отдельно по каждой страте.
    """
    generator = random.Random(seed)
    strata = defaultdict(list)
    for position, record in enumerate(records):
        band = 'уверенно' if record['confidence'] >= LOW_CONFIDENCE else 'спорно'
        if record['decided_by'] == 'rule':
            band = 'правило'
        strata[(record['model_class'], band)].append(position)

    per_stratum = max(1, size // max(len(strata), 1))
    chosen = []
    for key in sorted(strata):
        positions = strata[key]
        chosen.extend(generator.sample(positions, min(per_stratum, len(positions))))

    stratum_of = {position: key for key, positions in strata.items() for position in positions}
    rows = []
    for number, position in enumerate(sorted(chosen), 1):
        record = records[position]
        key = stratum_of[position]
        rows.append({
            'id': number,
            'predicted': 'chemistry' if record['category'] == 'formulas_chem' else 'не химия',
            'decided_by': record['decided_by'],
            'confidence': round(record['confidence'], 3),
            'true_class': '',
            'stratum': '{}/{}'.format(*key),
            # Сколько таких записей в корпусе: доли в выборке намеренно не
            # повторяют корпус, и без этого веса метрика по ней бессмысленна.
            'stratum_size': len(strata[key]),
            'doc_id': record['doc_id'],
            'page': record.get('page'),
            'latex': record['latex'],
            'context': (record.get('context') or '')[:300],
        })
    return rows


def _is_chemistry_label(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith('chem') or value in ('химия', 'х', 'c')


def score(sample_path: Path) -> dict:
    """Считает качество раскладки по размеченной выборке.

    Оценивается та же ось, по которой разложены каталоги: химия против всего
    остального. Физику от математики модель разделяет плохо, но в раскладке они
    лежат вместе, и на результат это не влияет.

    Доли в выборке не повторяют корпус — спорные записи набраны наравне с
    уверенными, иначе их в ней почти не было бы. Поэтому каждая строка входит в
    итог с весом своей страты, и точность получается корпусной, а не выборочной.
    """
    with sample_path.open(encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle) if row.get('true_class', '').strip()]
    if not rows:
        raise SystemExit(f'в {sample_path} не заполнена колонка true_class')

    counts = defaultdict(Counter)
    sizes = {}
    mistakes = []
    for row in rows:
        stratum = row['stratum']
        sizes[stratum] = int(row['stratum_size'])
        truth = _is_chemistry_label(row['true_class'])
        predicted = row['predicted'] == 'chemistry'

        counts[stratum]['sampled'] += 1
        counts[stratum]['correct'] += int(truth == predicted)
        counts[stratum]['true_chemistry'] += int(truth)
        counts[stratum]['predicted_chemistry'] += int(predicted)
        counts[stratum]['hit'] += int(truth and predicted)

        if truth != predicted:
            mistakes.append({
                'id': row['id'], 'stratum': stratum,
                'predicted': row['predicted'], 'true_class': row['true_class'],
                'latex': row['latex'][:140],
            })

    # Оценка по корпусу: доля внутри страты, умноженная на её размер.
    def weighted(field):
        return sum(sizes[name] * counts[name][field] / counts[name]['sampled']
                   for name in counts)

    corpus = sum(sizes.values())
    true_positive = weighted('hit')
    predicted_positive = weighted('predicted_chemistry')
    actual_positive = weighted('true_chemistry')
    precision = true_positive / max(predicted_positive, 1e-9)
    recall = true_positive / max(actual_positive, 1e-9)

    return {
        'labelled': len(rows),
        'corpus_formulas': corpus,
        'by_stratum': {
            name: {
                'sampled': counts[name]['sampled'],
                'in_corpus': sizes[name],
                'accuracy': round(counts[name]['correct'] / counts[name]['sampled'], 3),
                'ci95': _wilson(counts[name]['correct'], counts[name]['sampled']),
            }
            for name in sorted(counts)
        },
        'chemistry_bucket': {
            'estimated_size': round(predicted_positive),
            'true_chemistry_in_corpus': round(actual_positive),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        },
        'mistakes': mistakes,
    }


def _print_report(report: dict) -> None:
    print()
    print(f"формул: {report['formulas']}")
    print(f"  решено разбором формулы: {report['decided_by'].get('rule', 0)}")
    print(f"  решено моделью         : {report['decided_by'].get('model', 0)}")
    print('\nуверенность модели по классам:')
    for name, stats in report['confidence'].items():
        print(f"  {name:10s} медиана {stats['median']:.2f}  "
              f"p25 {stats['p25']:.2f}  p05 {stats['p05']:.2f}")
    low = report['low_confidence']
    print(f"\nответов ниже {low['threshold']}: {low['total']} "
          f"({low['share']:.1%}) — {low['by_class']}")
    reference = report['rule_as_reference']
    print(f"\nявная химия по разбору формулы: {reference['rule_positive']}")
    print(f"  модель узнала бы сама: {reference['model_agrees']} "
          f"({reference['model_recall_on_clear_chemistry']:.1%}, "
          f"ДИ95 {reference['model_recall_ci95']})")
    print(f"  назвала математикой  : {reference['model_says_math']}")
    print(f"  назвала физикой      : {reference['model_says_physics']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Оценка качества раскладки формул')
    parser.add_argument('command', choices=('report', 'sample', 'score'))
    parser.add_argument('--split', default=None, help='каталог раскладки (corpus_split)')
    parser.add_argument('--out', default=None, help='куда писать отчёт или выборку')
    parser.add_argument('--size', type=int, default=300, help='размер выборки для разметки')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=None, help='путь к formula_classifier.pkl')
    parser.add_argument('--cache', default=None, help='кэш ответов модели')
    args = parser.parse_args(argv)

    split_dir = Path(args.split or paths.ROOT / 'corpus_split')

    if args.command == 'score':
        result = score(Path(args.out or paths.REPORTS_DIR / 'split_audit_sample.csv'))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    report = analyse(split_dir, args.model, args.cache)
    records = report.pop('_records')

    if args.command == 'report':
        out = Path(args.out or paths.REPORTS_DIR / 'split_quality.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        _print_report(report)
        print(f'\nотчёт → {out}')
        return 0

    rows = build_sample(records, args.size, args.seed)
    out = Path(args.out or paths.REPORTS_DIR / 'split_audit_sample.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n{len(rows)} строк для разметки → {out}')
    print('заполните true_class (chemistry / math / physics), затем:')
    print('  python -m pdfscan.prepare.split_audit score')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
