"""Собирает англоязычный PDF для проверки языковой ветки парсера.

В корпусе только русские документы, поэтому английскую обработку не на чем
измерить. Образец повторяет типичный учебник: абзацы вперемешку с выключными
формулами по химии, физике и математике, набранными настоящим текстовым слоем
с индексами и степенями.
"""

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42  # шрифты как TrueType, с таблицей юникода

PAGE = [
    ('title', 'Chapter 2. Reduction of Iron Oxides in the Blast Furnace'),
    ('body', 'The reduction of iron ore proceeds stepwise as the burden descends'),
    ('body', 'through the shaft. Carbon monoxide acts as the principal reducing agent,'),
    ('body', 'and the overall reaction is written as follows:'),
    ('formula', r'$Fe_2O_3 + 3CO = 2Fe + 3CO_2$'),
    ('body', 'The direct reduction by solid carbon becomes significant above 1000 °C,'),
    ('body', 'where the Boudouard reaction shifts towards carbon monoxide:'),
    ('formula', r'$C + CO_2 = 2CO - 172\ kJ/mol$'),
    ('body', 'Desulphurisation takes place in the hearth, where lime reacts with'),
    ('body', 'ferrous sulphide dissolved in the metal:'),
    ('formula', r'$FeS + CaO + C = CaS + Fe + CO$'),
    ('body', 'The elastic response of the stand is described by Hooke law, which'),
    ('body', 'relates the stress to the strain through the elastic modulus:'),
    ('formula', r'$\sigma = E\,\varepsilon$'),
    ('body', 'The heat required to raise the temperature of the charge follows from'),
    ('body', 'the specific heat capacity of the material:'),
    ('formula', r'$Q = m\,c\,\Delta T$'),
    ('body', 'The yield of the process is defined as the ratio of the practical mass'),
    ('body', 'of the product to the theoretical mass computed from the equation:'),
    ('formula', r'$w = \frac{m_{pract}}{m_{theor}} \cdot 100\ \%$'),
    ('body', 'Substituting the numerical values gives the final estimate:'),
    ('formula', r'$\frac{6 \cdot 10^{8}}{143} \approx 4 \cdot 10^{6}$'),
    ('body', 'This value shows how much larger the nuclear energy release is when'),
    ('body', 'compared with the chemical combustion of the same amount of hydrogen.'),
]

STYLE = {
    'title': {'size': 15, 'weight': 'bold', 'x': 0.10, 'gap': 0.055},
    'body': {'size': 11, 'weight': 'normal', 'x': 0.10, 'gap': 0.033},
    'formula': {'size': 13, 'weight': 'normal', 'x': 0.42, 'gap': 0.055},
}


def build(path='english_sample.pdf'):
    figure = plt.figure(figsize=(8.27, 11.69))  # A4
    y = 0.93
    for kind, text in PAGE:
        style = STYLE[kind]
        if kind == 'formula':
            y -= 0.012
        figure.text(style['x'], y, text, fontsize=style['size'],
                    fontweight=style['weight'], va='top')
        y -= style['gap']

    figure.savefig(path)
    plt.close(figure)
    print(f'💾 {path}')


if __name__ == '__main__':
    build()
