# Notas da sessão de 2026-11-10: como ler o documento sucinto

Para reler antes de apresentar. Documento: https://claude.ai/artifact/FUd5qxRgDxHD5Vc8Ho7nF2

## O que o experimento revelou

| Comparação | Contexto longo | RAG BM25 | Diferença (RAG − contexto longo) | Veredito |
|---|---|---|---|---|
| H1 direta (MH-QA, 100 perguntas) | 42,0% | 56,0% | +14,0 pp [+2,0; +26,0] | Confirmada e reproduzida |
| H2 direta (cinco MCC, 500) | 77,0% | 76,8% | −0,2 pp [−2,8; +2,4] | Inconclusiva |
| H1 OAB (lei da matéria, 300) | 58,7% | 64,0% | +5,3 pp [+1,7; +9,0] | Confirmada |
| H2 OAB (provas antigas, 224) | 65,2% | 66,1% | +0,9 pp [−3,6; +5,4] | Inconclusiva |

- **H1 se sustenta:** para achar um trecho preciso, o RAG vence o contexto longo, nos dados do autor e
  na OAB (com o GPT-4o-mini).
- **H2 não apareceu:** para aprender com exemplos, o contexto longo não venceu o RAG em nenhuma etapa.
  Não foi contrariada: os dois empataram.
- **O RAG é bem mais barato:** na OAB, o contexto longo custou de 1,5 a 14,2 vezes mais.
- **Com o Haiku, a H1 da OAB empata** (73,7% contra 74,3%). A vantagem do RAG ali é só o custo.
- **Trocar o modelo pesou tanto quanto a memória** (exploratório): o Haiku sem memória (69,3%) fica
  acima do melhor GPT-4o-mini com RAG (64,0%).
- **Uma medida não reproduziu:** a média do contexto longo nas MCC deu 77,0%, contra 82,0 do artigo. A
  depuração descartou bug nosso. Sobram o modelo servido hoje e os dados publicados.

## Conceitos que perguntei

**Intervalo de 95%.** É a margem de incerteza da diferença RAG − contexto longo. O número antes dos
colchetes é a diferença medida; os colchetes dizem até onde ela poderia ir com outras perguntas do mesmo
tipo. Ninguém escolhe o intervalo: ele sai de uma conta sobre os dados.

**O zero.** É o empate. Diferença positiva: o RAG acertou mais. Negativa: o contexto longo acertou mais.

- Intervalo sem zero (`[+2,0; +26,0]`): mesmo no pior caso o RAG ganha. Há veredito.
- Intervalo com zero (`[−2,8; +2,4]`): pode ganhar um, o outro, ou empatar. Inconclusiva.

**O que foi escolha nossa, antes de rodar:** o nível de 95% (convenção), 10.000 sorteios, a semente 42 e
a regra "sem zero dá veredito, com zero é inconclusiva".

**O artigo não tem intervalo.** Ele reporta só o número de acerto de cada agente (procurei "bootstrap" e
"confidence interval" no PDF e não há). O intervalo é acréscimo nosso. "Reproduz" quer dizer que o número
pontual do artigo cai dentro do nosso intervalo.

**Bootstrap.** Como o intervalo é calculado:

1. Cada agente respondeu as perguntas **uma vez só** (é a parte que custa dinheiro). Fica uma tabela:
   por pergunta, se cada agente acertou.
2. O programa sorteia N linhas dessa tabela **com reposição** (a pergunta sorteada volta para o saco;
   algumas saem repetidas, outras ficam de fora) e recalcula a diferença.
3. Repete 10.000 vezes, ordena as diferenças e fica com os 95% do meio.

Nenhuma pergunta volta ao modelo: é só aritmética sobre as respostas já pagas. O intervalo mede o quanto o
resultado balança se as perguntas fossem outras. Ele não mede a variação da temperatura 0,7, porque só
houve uma rodada por configuração.

**Pareado.** Os dois agentes responderam as mesmas perguntas, e o sorteio leva a linha inteira (o
resultado dos dois naquela pergunta). Perguntas que os dois acertam ou os dois erram se cancelam; a
diferença depende só daquelas em que um acertou e o outro errou. Isso deixa o intervalo mais estreito.

**MCC (classificação multiclasse).** Tarefa da H2 direta, "aprender com exemplos". A memória recebe
milhares de frases rotuladas com números ("frase → rótulo 43"); a pergunta é uma frase nova, e a resposta
é o número do rótulo. São cinco datasets de 100 perguntas: BANKING77, CLINC150, NLU, TREC coarse e TREC
fine.

**MH-QA (perguntas multi-hop sobre documentos).** Tarefa da H1 direta, "achar um trecho preciso". A
memória é uma pilha de documentos de cerca de 421 mil tokens; a resposta exige juntar dois ou mais
trechos. São 100 perguntas, do dataset do autor; nós rodamos os três agentes sobre elas.

**Equivalente na OAB.** A H1 da OAB: 300 questões de 2018 a 2023, com a lei da matéria como memória. Não
é multi-hop e não tem coluna "Artigo" (o autor nunca testou a OAB), então só tem veredito, sem teste de
reprodução.

**RAG.** Dois passos: um buscador escolhe os 10 trechos mais relevantes da memória, e o modelo lê só
esses.

**BM25.** O buscador por palavras em comum. Palavra rara vale mais; repetição e tamanho do trecho são
descontados. Não entende significado ("carro" e "automóvel" não casam). Roda local, de graça.

**RAG por embedding.** O buscador por significado: transforma pergunta e trechos em vetores e busca os
mais próximos. Pega sinônimos, mas custa uma chamada de API.

**Indexar.** Preparar a memória inteira para a busca: cortar em trechos e catalogar cada um. O índice fica
fora do modelo, sem limite de janela. Só os 10 trechos escolhidos passam pela janela.

## A ressalva mais importante: contexto longo não é contexto inteiro

O agente do autor corta a memória para caber na janela e guarda o fim: cerca de 124 mil tokens com o
GPT-4o-mini. O BM25 indexa tudo.

| | Contexto longo | RAG |
|---|---|---|
| O que alcança | só os últimos 124 mil tokens | a memória inteira |
| O que o modelo lê | tudo o que alcança | só 10 trechos |
| Risco | a resposta estar na parte cortada | a busca não trazer o trecho certo |

- **No MH-QA, pesa muito:** o modelo vê menos de um terço dos 421 mil tokens. O resultado mostra "o RAG
  vence quando o contexto não cabe na janela" com mais segurança do que "o RAG lê melhor".
- **Na OAB, pesa menos:** o corte atinge 7 das 15 leis (159 das 300 questões), mas o BM25 vence tanto nas
  leis cortadas (+5,0 pp) quanto nas inteiras (+5,7 pp). O corte não parece explicar a vitória
  (exploratório). Além disso, o modelo já sabe direito: sem memória, acerta 54,3%.
- **O RAG por embedding do autor também esquece:** acima de cerca de 285 mil tokens, descarta os trechos
  mais antigos. No MH-QA, indexa só os últimos dois terços. Ele fez 45,0% e não se separou do contexto
  longo; a vitória ali é do BM25, não do RAG em geral.
