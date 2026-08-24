"""Проставляет предложенные метки в harvested_dataset.csv.

Метки остаются непринятыми (needs_review = 1), пока их не подтвердит человек:
в обучение попадают только строки с нулём в этой колонке.
"""

import pandas as pd

from pdfscan.paths import HARVESTED_DATASET

# Ключ — начало формулы, значения — предлагаемая метка и её обоснование.
PROPOSALS = [
    ('ΔH° = 2Δ_{f}H°', 'chemistry',
     'энтальпия реакции по закону Гесса, аргументы — химические соединения'),
    ('w=\\frac{m_{практ}}{m_{теор}}', 'chemistry',
     'выход реакции: отношение практической массы продукта к теоретической'),
    ('n_{NH4VO3} =\\frac{m}{M}', 'chemistry',
     'количество вещества через молярную массу'),
    ('m_{теор} =nM', 'chemistry',
     'масса по количеству вещества и выход реакции в процентах'),
    ('H_{2} + \\frac{1}{2}O_{2}', 'chemistry',
     'уравнение горения водорода; грамматика не поймала из-за дробного коэффициента'),
    ('E = 0,0063', 'physics',
     'ИСПРАВЛЕНИЕ модели: E=mc² по дефекту массы, ядерная физика, а не химия'),
    ('\\frac{6 · 10^{8}}{143}', 'math',
     'ИСПРАВЛЕНИЕ модели: чистая арифметика — во сколько раз одна энергия больше другой'),
    ('σ_{m} - гидростатическое', 'physics',
     'определение гидростатического напряжения'),
    ('Ш = \\frac{M_{шл}}{M_{чуг}}', 'chemistry',
     'выход шлака как балансовое отношение масс, по смыслу рядом с выходом реакции'),
    ('−ln(η)+ln(β)+ln(λ)=0', 'physics',
     'связь логарифмов коэффициентов деформации; величины физические'),
    ('или за n проходов Δh_{Σ}', 'physics',
     'абсолютное обжатие за несколько проходов прокатки'),
    ('ε_{Σ}=\\frac{h_{0} −h_{n}}{h_{0}}', 'physics',
     'относительная деформация и абсолютное уширение'),
    ('δ = h_{1} – s_{0}', 'physics',
     'пружина прокатной клети'),
    ('δ=s_{1}+\\frac{P}{tgθ}', 'physics',
     'пружина клети через модуль жесткости'),
    ('h_{1}=s_{0}+\\frac{P}{M_{k}}', 'physics',
     'толщина полосы на выходе через модуль жесткости клети'),
    ('; ; f_{3}=\\frac{1,2p_{1}l_{1}}', 'physics',
     'прогиб от поперечных сил; запись рваная, но величины механические'),
    ('f_{1}+f_{2}+f_{3}=1,2', 'physics',
     'суммарная упругая деформация станины'),
]


def main(path=HARVESTED_DATASET):
    frame = pd.read_csv(path)
    frame['proposed_reason'] = frame.get('proposed_reason', '')

    matched = set()
    for prefix, label, reason in PROPOSALS:
        mask = frame['text'].astype(str).str.startswith(prefix)
        if not mask.any():
            print(f'⚠️ не найдено: {prefix!r}')
            continue
        changed = frame.loc[mask, 'class'].ne(label).any()
        frame.loc[mask, 'class'] = label
        frame.loc[mask, 'label_source'] = 'предложено'
        frame.loc[mask, 'proposed_reason'] = reason
        matched.update(frame.index[mask])
        mark = '✏️' if changed else '  '
        print(f'{mark} {label:<9} {reason}')

    stale = frame.index[(frame['needs_review'] == 1) & (~frame.index.isin(matched))]
    if len(stale):
        print(f'\n⚠️ без предложения осталось строк: {len(stale)}')
        for index in stale:
            print(f'    {str(frame.loc[index, "text"])[:80]}')

    frame.to_csv(path, index=False)
    print(f'\n💾 {path}: предложено меток {len(matched)} из {len(frame)}')
    print('Метки не вступят в силу, пока в needs_review не будет 0.')


if __name__ == '__main__':
    main()
