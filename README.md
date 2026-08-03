# sub2k-nn (Mais leve que TensorFlow Lite Micro)

**Reconhecimento de dígitos manuscritos com uma rede neural quantizada (int8) rodando dentro de um Arduino Uno (2KB de RAM) - 96,7% de acurácia, 1,19KB de flash, ~18,5% de RAM usada, sem nenhuma biblioteca de ML no firmware.**

O problema que isso resolve: rodar uma rede neural "de verdade" num microcontrolador de 8 bits normalmente significa depender de frameworks pesados (TensorFlow Lite Micro, uTensor) ou aceitar que ponto flutuante vai comer a RAM escassa. O `sub2k-nn` treina a rede inteira em Python/float, quantiza os pesos pra `int8` uma única vez, e o firmware só faz multiplicação e soma em inteiros contra uma tabela fixa em `PROGMEM` - sem `float`, sem framework, sem alocação dinâmica.

Faz parte da mesma série de experimentos de restrição extrema em hardware do ecossistema **DevSoft JARVIS AI**, ao lado de [`sub2k-intent`](../sub2k-intent), [`sub2k-face`](../sub2k-face), [`sub2k-kws`](../sub2k-kws) e [`uno-nvscript`](../uno-nvscript). Junto com o `sub2k-intent` (hashing) e o `sub2k-face` (projeção PCA), fecha o trio das três famílias clássicas de classificação sob a mesma restrição de 2KB: **hashing**, **projeção linear** e **rede neural**.

---

## A ideia central

- **O treino fica no PC.** A rede aprende em float, com todo o ferramental normal (`scikit-learn`), sobre um dataset real de dígitos manuscritos 8×8 - o Uno nunca treina nada, só executa.
- **O comportamento vira uma tabela de pesos, não código novo.** Depois de treinada, a rede é quantizada e exportada como `nn_table.h`, incluído direto no firmware via `PROGMEM` - trocar o que a rede reconhece é gerar uma tabela nova, não reescrever a lógica de inferência.
- **Decisão em aritmética inteira.** O Uno só faz `int8 × int8` acumulado em `int32`, ReLU e `argmax` - sem `float`, sem `sqrt()`, sem biblioteca de ML.

## Arquitetura da rede (64 → 16 → 10)

| Camada | Tamanho | O que faz |
|---|---|---|
| Entrada | 64 | pixels de uma imagem 8×8 em escala de cinza (valores 0-16) |
| Oculta | 16 | `ReLU(entrada @ W1 + b1)` |
| Saída | 10 | `entrada_oculta @ W2 + b2`, uma pontuação por dígito (0-9) |

A camada oculta e a de saída somam **1.184 pesos** (quantizados em `int8_t`) **+ 26 biases** (`int8_t` na camada oculta, `int16_t` na de saída - cada bias usa o menor tipo que cabe nos valores reais do treino) - a tabela inteira cabe em **1,19KB de flash**.

## Arquitetura (treino no PC, inferência no Uno)

```
 scikit-learn (PC)                         Serial              Arduino Uno
┌─────────────────────────────┐    ┌──────────────────────┐   ┌─────────────────────────┐
│ load_digits() + MLPClassifier│    │ [0xAA] uint8[64]      │──▶│ camada 1: int32 += px*W1 │
│ quantização W1,W2,b1,b2 int8│───▶│ [checksum]            │   │ ReLU, camada 2, argmax   │
└─────────────────────────────┘    └──────────────────────┘   └─────────────────────────┘
                                                                          │
                                                                          ▼
                                                        "PRED=<dígito> SCORES=<10 inteiros>"
```

O treino roda uma vez (ou toda vez que o dataset mudar); a inferência no Uno é o que roda continuamente.

## Números reais

| Métrica | Valor |
|---|---|
| Acurácia do modelo em float (referência) | 96,4% |
| Acurácia do modelo quantizado (int8, aritmética inteira) | **96,7%** |
| Perda de acurácia por causa da quantização | **-0,3 pp** (na prática, nula) |
| Tamanho da tabela de pesos em flash | **1.220 bytes (1,19 KB)** - biases em `int8_t`/`int16_t` em vez de `int32_t` |
| RAM livre em runtime, medida com `freeMemory()` no hardware real | **1.669 B livres de 2.048 B totais (~18,5% usado, mais de 80% livre)** |
| Tempo de inferência isolado, medido com `micros()` no hardware real | **~4,87 ms** por predição |
| Margem de segurança contra overflow do acumulador int32 | 8,1x abaixo do limite |

A quantização não custou quase nada de acurácia - e a matriz de confusão mostra só um padrão de erro sistemático: o dígito **8 sendo confundido com 1** em 3 de 360 amostras de teste, um erro plausível mesmo pra um classificador maior, já que certas caligrafias de "8" ficam visualmente parecidas com "1" numa grade de só 8×8 pixels.

## Como usar

**0. Instale as dependências do PC:**
```bash
pip install -r requirements.txt --break-system-packages
```

**1. Treine e valide a rede (sem precisar de hardware):**
```bash
python3 pc/train_nn.py     # treina, quantiza, exporta firmware/sub2k_nn/nn_table.h
python3 pc/test_nn.py      # matriz de confusão + checagem de overflow do acumulador
```

**2. Grave o firmware no Arduino:**
Abra `firmware/sub2k_nn/sub2k_nn.ino` no Arduino IDE - o `nn_table.h` gerado no passo 1 precisa estar na mesma pasta - selecione a placa "Arduino Uno" e a porta COM certa, e grave.

**3. Teste no hardware real:**
```bash
python3 pc/predict_live.py COM5          # Windows
python3 pc/predict_live.py /dev/ttyUSB0  # Linux
```
Pega uma amostra aleatória (ou um índice específico com `--index`) do dataset, envia pro Arduino, e compara a predição do hardware com o rótulo real.

## Protocolo (Serial, 9600 baud)

```
PC -> Uno: [0xAA] [uint8[64]] [checksum: 1 byte]
             sync    pixels     soma dos 64 bytes, mod 256

Uno -> PC: "PRED=<dígito> SCORES=<10 inteiros>\n" | "ERR\n"
```

O framing (sync byte + checksum) segue o mesmo padrão usado no `sub2k-face` e no `sub2k-kws` - os projetos da série falam Serial do mesmo jeito, então dá pra reaproveisar o mesmo raciocínio de resincronização: se o checksum não bater, o Uno responde `"ERR"` e descarta o pacote, em vez de rodar a inferência sobre um vetor corrompido.

## Por que a aritmética não estoura

O acumulador é `int32_t`. No pior caso teórico (todo peso no valor máximo, toda entrada no valor máximo - o que nunca acontece de verdade, mas serve como garantia formal), a camada 2 chega a ~264 milhões, contra um limite de ~2,1 bilhões do `int32_t`. Isso dá uma margem de **8,1x** - folga suficiente mesmo se o modelo for retreinado com dados diferentes. `pc/test_nn.py` calcula esse limite automaticamente a cada treino, então qualquer mudança de arquitetura que aproxime demais do limite aparece como aviso, não como bug silencioso em produção.

## Segurança e limites conhecidos

- **Checksum inválido** → o Uno responde `"ERR"` e descarta o pacote inteiro, nunca roda inferência sobre um vetor não verificado.
- **Dataset fixo de 8×8** - trocar o que a rede reconhece (outra coisa que não dígitos manuscritos) exige retreinar do zero com `pc/train_nn.py`; não há "aprendizado incremental" em runtime, ao contrário do `sub2k-intent` (onde adicionar um comando é só editar um JSON).
- **Sem calibração de confiança** - os `SCORES` retornados são a pontuação bruta do `argmax`, não uma probabilidade calibrada; um score "vencedor" baixo ainda é reportado como predição, sem limiar de rejeição embutido no firmware (diferente do `sub2k-face`/`sub2k-kws`, que rejeitam abaixo de um threshold).
- **Overflow do acumulador** - verificado formalmente (ver seção acima), com margem de 8,1x; ainda assim, uma arquitetura muito maior (mais neurônios ocultos) reduziria essa margem e merece novo cálculo.

## Estrutura do projeto

```
sub2k-nn/
├── pc/
│   ├── train_nn.py         # treina, quantiza e exporta a tabela de pesos
│   ├── nn_sim.py            # simulador standalone (mesma aritmética do .ino)
│   ├── test_nn.py            # matriz de confusão + checagem de overflow
│   ├── predict_live.py        # envia uma amostra pro Arduino real via Serial
│   └── nn_table.npz            # pesos quantizados (gerado pelo treino)
│
├── firmware/
│   └── sub2k_nn/
│       ├── sub2k_nn.ino     # firmware - grava no Arduino uma vez
│       └── nn_table.h        # tabela de pesos gerada automaticamente
│
├── requirements.txt
└── README.md
```

## Requisitos

- Arduino Uno (ATmega328P) ou qualquer AVR compatível
- Arduino IDE (pra gravar o firmware)
- Python 3.10+ no PC, com `numpy` e `scikit-learn` (`pip install -r requirements.txt`)
- `pyserial` só se for testar no hardware real (já incluído no `requirements.txt`)

## Status

- [x] Arquitetura da rede definida (64 → 16 → 10) e treino com dataset real
- [x] Quantização int8 validada (perda de acurácia: -0,3pp, dentro do ruído)
- [x] Matriz de confusão e checagem formal de overflow do acumulador
- [x] Firmware `.ino` com framing de protocolo (sync byte + checksum), alinhado com sub2k-face e sub2k-kws
- [x] Teste ponta-a-ponta com hardware real via `predict_live.py`
- [x] Biases otimizados pra `int8_t`/`int16_t` (em vez de `int32_t`), com checagem de faixa no `train_nn.py` - economia de ~70 bytes de flash sem custo de RAM, velocidade ou acurácia
- [ ] Rejeição por threshold de confiança (hoje sempre retorna o argmax, mesmo com score baixo)

## O "trio" de técnicas de classificação sob 2KB

| Projeto | Técnica | Tipo de entrada | Compressão | Acurácia/Precisão |
|---|---|---|---|---|
| [`sub2k-intent`](../sub2k-intent) | Hashing de trigramas | Texto | ~1,5KB de flash pra 47 frases | 88% (15/17 no teste com casos difíceis) |
| [`sub2k-face`](../sub2k-face) | Projeção linear (PCA/Eigenfaces) | Imagem | ~44 bytes por identidade cadastrada | 100% em dados sintéticos validados; fotos reais ainda pendentes |
| **`sub2k-nn`** | Rede neural (MLP 2 camadas) | Imagem | **1,19KB** pra 64→16→10 | **96,7%** em dataset real |

O `sub2k-nn` é o mais "clássico" das três abordagens (é literalmente uma rede neural, só que pequena e quantizada), e também o único treinado e validado inteiramente sobre dados reais (não sintéticos) desde o início.

## Direção futura: combinar com uno-nvscript

Hoje a tabela de pesos (`nn_table.h`) é fixa em `PROGMEM`, gravada junto com o sketch. Uma direção natural, na mesma linha do `sub2k-face`, é permitir retreinar/trocar a tabela sem recompilar - enviando os pesos quantizados por Serial e gravando em EEPROM, aproveitando a mesma filosofia de dado-em-vez-de-código do [`uno-nvscript`](../uno-nvscript). **Essa integração ainda não está implementada** - é um próximo passo natural, não algo que já roda.

## Sobre este projeto

Desenvolvido como parte de uma série de experimentos de engenharia de restrição extrema em hardware ultra-limitado, no contexto do ecossistema de automação/IoT do **DevSoft JARVIS AI**.

**Autor:** Bruno Nunes da Silva (criador do DevSoft JARVIS AI)<br>
**Conheça o DevSoft JARVIS AI:** https://devsoft-ai.webnode.page/<br>
**Canal no YouTube:** https://www.youtube.com/@devsoftai5538

## Licença

MIT - use, modifique e distribua livremente, mantendo os créditos de autoria.
