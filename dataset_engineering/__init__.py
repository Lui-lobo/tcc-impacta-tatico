"""Fluxo de engenharia de dados do dataset CWRU.

Quatro etapas encadeadas, do .mat bruto ao dataset curado:

    1. leitura      - le e demonstra o dataset como ele e
    2. diagnostico  - mede o ANTES e lista os defeitos encontrados
    3. curadoria    - aplica as tratativas e materializa data_curado/
    4. comparacao   - mede o DEPOIS e confronta com o ANTES

    python -m dataset_engineering.executar
"""
