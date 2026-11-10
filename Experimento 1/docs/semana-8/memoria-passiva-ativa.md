# Memória passiva e memória ativa com LLM: dois exemplos comparáveis

**Residência em IA (UFG) · Semana 8 · Pedro Coelho**

A entrega da semana 8 é implementar e documentar um exemplo de memória passiva e um de memória ativa para um LLM. Este documento segue o framework de três passos fechado na semana 7 (diagnóstico, desenho, verificação) e usa o vocabulário dos artigos lidos.

Os dois exemplos compartilham tudo: o mesmo modelo (`claude-sonnet-5-5`), o mesmo system prompt base, a mesma janela de mensagens recentes, o mesmo banco SQLite e a mesma conversa. Eles diferem só em **quem controla a leitura e a escrita da memória**. No exemplo passivo é o código; no ativo, o próprio modelo. Assim a comparação isola a variável que a semana 7 destacou.

- Exemplo 1, memória passiva: [`memoria/passiva.py`](../../memoria/passiva.py)
- Exemplo 2, memória ativa: [`memoria/ativa.py`](../../memoria/ativa.py)
- Verificação: [`memoria/avaliacao/`](../../memoria/avaliacao/), com os resultados da rodada em [`resultados/20261027-231357/`](../../resultados/20261027-231357/)

**O resultado em poucas linhas.** Na única rodada da verificação, com 17 perguntas, o contexto completo e a memória ativa acertaram 17/17, a passiva 16/17 e a configuração sem memória 4/17 (só as abstenções). A ativa foi a mais barata das três configurações que atravessam sessões (US$ 1,08). A passiva foi a mais cara de todas (US$ 2,73): o resumo, que devia ter no máximo 400 palavras, chegou a 1.171, porque o limite era só uma instrução no prompt e o modelo não a seguiu. A hipótese escrita antes da rodada acertou o placar em linhas gerais e a posição da ativa no custo, entre o piso e o teto, mas errou o custo da passiva, que ficou acima do contexto completo.

## 1. Diagnóstico

O cenário é o de um assistente pessoal que conversa com a mesma usuária em sessões separadas por dias ou semanas. São cinco sessões, de 2026-03-02 a 2026-04-27, e uma sessão final de perguntas em 2026-05-11. A janela recente tem as últimas K = 6 mensagens da sessão atual (três turnos) e zera quando a sessão muda.

O primeiro passo do framework pergunta se o problema precisa mesmo de memória externa. As perguntas da semana 7, aplicadas ao cenário:

| Pergunta da semana 7 | Neste cenário | Evidência |
|---|---|---|
| O histórico cabe na janela? | Sim, com folga. | O contexto completo, que reenvia a conversa inteira, teve em média 10.643 tokens de entrada por chamada (808.855 em 76 chamadas). O LongMemEval começa em ~115K tokens (Wu et al., 2025). |
| O modelo usa bem o que está lá? | Sim, nesta escala. | O contexto completo acertou 17/17. A queda de 30% a 66% ao ler o histórico inteiro (Wu et al., 2025, §3.4) aparece com históricos muito maiores que este. |
| Quanto custa? | Cresce a cada turno. | Para ver tudo, o contexto completo reenvia tudo em toda chamada: 808.855 tokens de entrada, mais de 15 vezes os 52.664 da configuração sem memória, que vê só a janela. Num assistente de verdade o histórico cresce sem teto, e esse custo cresce junto. |
| A informação muda? | Sim. | Três fatos mudam entre sessões: o emprego (Cerrado Logística → Agrotech Planalto, em 2026-04-13), o bairro (Setor Bueno → Jardim América, em 2026-04-27) e o esporte (natação → beach tennis, em 2026-04-27). |

Há ainda um motivo que não depende do tamanho: **as sessões são separadas**. A janela zera na troca de sessão, então sem memória nada passa de uma sessão para a outra. A configuração sem memória mostra isso na tabela de resultados: 0/13 em tudo que não é abstenção.

A tabela "Quando usar cada abordagem" da semana 7 manda ler de cima para baixo e parar na primeira linha que resolve o caso. Aplicada a este cenário:

| Linha da tabela da semana 7 | Resolve este cenário? | Onde aparece aqui |
|---|---|---|
| Só LLM com workflow (prompt e passos fixos, sem estado entre sessões) | Não. O fato é privado e foi dito em outra sessão: os pesos não o têm. | baseline `sem_memoria` (o piso) |
| Contexto longo (histórico inteiro na janela) | Nesta miniatura, sim: o histórico cabe e o acerto foi 17/17. O sinal de que não basta, "o histórico cresce sem teto", é o caso de um assistente real; aqui ele aparece só no custo. | baseline `contexto_completo` (o teto) |
| RAG (índice externo, só leitura) | Não. Os fatos mudam e algo precisa ser esquecido. | nenhum |
| Memória acoplada (camada persistente que extrai e atualiza fatos por pipeline: Mem0, Zep) | Sim. É preciso lembrar da usuária entre sessões, e esses fatos mudam. | **exemplo 1, passiva**, na versão mais simples possível: um resumo em vez de um banco de fatos |
| Agente gerindo a memória (o LLM decide ler, escrever e reorganizar: MemGPT, A-MEM, MIRIX) | Só se não desse para prever o que guardar. Aqui dá: são fatos sobre a usuária. | **exemplo 2, ativa** |

Leitura honesta da tabela: nesta escala o contexto longo bastaria, e a linha da memória acoplada vem antes da linha do agente. Os dois exemplos estão aqui porque a entrega pede um de cada e porque a comparação entre eles é o que ensina. A miniatura simula o problema real, um histórico sem teto, apenas pelo custo por turno.

## 2. Exemplo 1: memória passiva por sumarização incremental

### Onde ele cai no vocabulário da semana 7

- **Política de controle: heurística.** É a primeira linha da tabela "Política de controle" da semana 7: regras fixas, do tipo "resumir a cada n passos", aqui com n = 1. O parente mais próximo nas leituras é a fila FIFO do MemGPT, que despeja as mensagens antigas com resumo recursivo (Packer et al., 2023, §2.2). O modelo que conversa com a usuária nunca sabe que existe memória.
- **Tipo de memória: semântica, com uma linha do tempo episódica comprimida.** O resumo S tem quatro seções. "Sobre o usuário" e "Fatos e decisões" são memória semântica: fatos tirados do contexto em que apareceram (Sumers et al., 2024, §4.1). "Linha do tempo" guarda eventos com data, que é memória episódica comprimida, como a semana 7 classifica os resumos de sessão do LoCoMo (Maharana et al., 2024). "Pendências" guarda o que ficou em aberto.
- **Ciclo escrever → gerenciar → ler (Du, 2026, §2).** Ler: o código carrega S e o injeta inteiro no system prompt, sem busca. Escrever e gerenciar acontecem na mesma chamada: o resumidor recebe S e o turno que acabou de acontecer e devolve S', acrescentando o que é novo, substituindo o que foi contradito e, pela instrução, comprimindo o que passar do limite. É a atualização U de Du feita inteira por um LLM, a cada turno.
- **Não é RAG.** Não há índice, consulta nem recuperação: S entra inteiro em toda chamada. É **context compression**, uma das três linhas de memória de longo prazo que o LongMemEval descreve, ao lado do long-context e dos módulos de memória diferenciáveis: condensar o histórico numa representação compacta (Wu et al., 2025).

### Pipeline de um turno

Copiado do docstring de `passiva.py`:

```text
mensagem do usuário
   |
   +--> 1. carrega o resumo atual S da conversa (SQLite)
   |
   +--> 2. monta o prompt: system base + data + bloco "Resumo do que você sabe..." + S
   |       + últimas K mensagens da sessão + mensagem nova
   |       +--> modelo responde (sem ferramentas)
   |
   +--> 3. chamada ao resumidor: recebe S + (mensagem, resposta) + data da sessão
           +--> devolve S' e grava no SQLite (S' passa a ser o S do próximo turno)
```

Cada turno custa duas chamadas à API: a resposta e o resumidor. É por isso que a passiva tem o dobro de chamadas das baselines na tabela de resultados.

### O prompt do resumidor

Copiado de `passiva.py`. `{limite}` recebe `LIMITE_PALAVRAS_RESUMO` (400) e `{data}` recebe a data da sessão.

```python
PROMPT_RESUMIDOR = """\
Você mantém o resumo de uma conversa entre um usuário e um assistente. Esse resumo é tudo o que o \
assistente vai lembrar da conversa nos próximos turnos, então ele precisa guardar o que importa.

O resumo tem sempre estas quatro seções, nesta ordem:

## Sobre o usuário
## Fatos e decisões
## Pendências
## Linha do tempo

Você recebe o RESUMO ATUAL e o NOVO TURNO (a mensagem do usuário e a resposta do assistente). \
Devolva o resumo atualizado. EDITE o resumo atual, não o reescreva do zero:
- mantenha o que continua válido;
- acrescente o que o novo turno trouxe de novo;
- substitua o que o novo turno contradisse ou corrigiu;
- registre na seção Linha do tempo o que tem data, no formato `AAAA-MM-DD: evento`. Converta datas \
relativas ("ontem", "semana que vem") em datas completas a partir de hoje.
Seção sem conteúdo fica com `(nada ainda)`.

O resumo não pode passar de {limite} palavras. Se passar, comprima o que for menos importante.

Hoje é {data}.

Responda SOMENTE com o resumo em Markdown, sem comentários, sem explicações e sem cercas de código."""
```

### Decisões de desenho

- **Estrutura fixa e instrução de editar, não reescrever.** É a mitigação do colapso de contexto descrito no ACE, em que reescrever a memória inteira a cada passo vai apagando o conteúdo; no Dynamic Cheatsheet, o contexto caiu de 18.282 para 122 tokens (Zhang et al., 2026).
- **Limite de 400 palavras, só como instrução.** Nada no código conta palavras. Por desenho (spec §5.1), o limite fica no prompt do resumidor. A seção 4 mostra que o modelo não o respeitou.
- **Resumo a cada turno.** Resumir só quando a janela estoura sairia mais barato, mas seria outra variante, com outro comportamento.
- **Falha não apaga a memória.** S' só é gravado se a saída do resumidor tiver texto e não começar com `[erro:`. O banco guarda só o S atual; a evolução de S é copiada pelo runner da verificação ao fim de cada sessão.
- **Erro de API vira texto.** Um turno que falha recebe a resposta literal `[erro: <motivo>]`, que entra na janela como resposta do assistente; o resumidor não roda nesse turno.

### S real, da rodada de verificação

O runner copia S ao fim de cada sessão para `passiva.json`. Abaixo, S depois da sessão 1 (2026-03-02, 700 palavras) e depois da sessão 5 (2026-04-27, 1.171 palavras), copiados do JSON sem edição.

S depois da sessão 1:

```markdown
## Sobre o usuário
- Analista de dados na Cerrado Logística.
- Segundas-feiras são sempre o dia mais corrido no trabalho dele.
- Mora em Goiânia; está sofrendo com o calor e não tem ar-condicionado (ou prefere dormir sem ele).
- Tem um cachorro chamado Paçoca.
- O chefe costuma marcar reuniões; tem uma às 8h de amanhã.
- Faz natação três vezes por semana, às 6h da manhã; acordar cedo está sendo um sofrimento. Os dias de treino ainda não foram informados.
- Está pensando em comprar uma cafeteira (italiana ou de filtro); ainda não informou quanto café bebe nem o orçamento.
- Quer lanche prático para levar ao trabalho; restrições alimentares e acesso a geladeira no trabalho ainda não informados.

## Fatos e decisões
- Hoje (2026-03-02) é segunda-feira, dia corrido para o usuário.
- Organização da semana: o assistente sugeriu três grupos (urgente hoje, importante da semana, pode esperar); 3 prioridades por dia; segunda mais leve; 15 min no início do dia para revisar a lista e 15 no fim para ajustar o dia seguinte.
- Dormir no calor sem ar-condicionado: ventilação cruzada à noite e cortinas fechadas de dia; ventilador para parede/teto ou com bacia de gelo; banho morno antes de deitar; cama leve de algodão; água gelada ou pano úmido nos pulsos e pescoço; evitar refeição pesada e álcool à noite; hidratar-se ao longo do dia.
- Paçoca roeu o pé da mesa da sala hoje. Causas comuns: tédio, ansiedade, energia sobrando ou dentição (filhote). Sugestões: brinquedos de roer e elogiar o uso; passear ou brincar antes de sair; spray amargo ou barreira no pé da mesa; sem bronca depois do fato, redirecionar na hora.
- Mancha de café em camisa branca: enxaguar com água fria pelo avesso; detergente neutro ou sabão de coco por 10 a 15 min; bicarbonato ou vinagre branco se sobrar marca; manchas antigas, molho com oxigenada ou removedor tipo Vanish; conferir antes de secar. Evitar cloro em tecido com elastano ou poliéster.
- Média x mediana: média = soma dividida pela quantidade; mediana = valor central dos dados ordenados. Exemplo: 2, 2, 3, 3, 40 → média 10, mediana 3. A mediana resiste a outliers.
- Reunião das 8h de amanhã: dormir cedo, deixar roupa, bolsa e pauta prontas, luz natural ao acordar, água, café da manhã leve com proteína e fruta, café com moderação, alongar 5 min, banho morno ou frio.
- Acordar às 6h para natação: horário fixo de dormir e acordar (inclusive fim de semana); 7 a 8 h de sono; deixar mochila, maiô, touca, toalha e roupa prontos na noite anterior; alarme longe da cama; luz ao acordar; lanche leve antes (banana ou pão); água ao acordar; lembrar que a natação dá energia.
- Cafeteira: italiana (moka) = café forte, barata, durável, sem eletricidade, mas exige atenção no fogão e faz porções pequenas. Filtro elétrica = prática, café mais suave e em maior quantidade, ocupa espaço, precisa de energia e papel filtro. O assistente sugeriu que a elétrica tende a ser mais cômoda pelas manhãs puxadas, só palpite; sem decisão.
- Lanches para o trabalho (sugeridos): frutas sem geladeira (banana, maçã, tangerina), castanhas, sanduíche de pão integral (com calor, preferir pasta de amendoim), barrinha com pouco açúcar, iogurte natural com aveia e ovo cozido (precisam de geladeira), pipoca de panela, biscoito integral. Dica: combinar proteína com fruta ou carboidrato.

## Pendências
- O assistente ofereceu montar a divisão das tarefas da semana (antes, listar e priorizar as do dia). O usuário ainda não passou as tarefas.
- O assistente ofereceu uma rotina para esta noite/antes de dormir; o usuário ainda não respondeu.
- O assistente perguntou a idade do Paçoca; sem resposta.
- O assistente perguntou se amanhã é dia de natação (se for, com a reunião às 8h, dormir ainda mais cedo); sem resposta.
- Cafeteira: usuário ainda não decidiu; falta saber quanto café bebe, se mora com mais gente e o orçamento.
- Lanche: o assistente perguntou sobre restrições alimentares e geladeira no trabalho para ajustar as sugestões; sem resposta.

## Linha do tempo
- 2026-03-02: Paçoca roeu o pé da mesa da sala.
- 2026-03-03: reunião com o chefe às 8h da manhã.
```

S depois da sessão 5:

```markdown
## Sobre o usuário
- Cientista de dados na Agrotech Planalto (começou em 2026-04-13); antes, analista na Cerrado Logística. Usa Python. O chefe antigo marcava reuniões; ela quer recusar as desnecessárias.
- Mulher; mudou-se no fim de semana (2026-04-25/26) para o Jardim América, em Goiânia; ainda com caixas por abrir (pediu ordem para desencaixotar). Antes morava no Setor Bueno, em apartamento com pouca luz, sem ar-condicionado (sofria com o calor) e com obra no prédio vizinho. Não se sabe se o novo apê tem mais luz, nem se tem calor, barulho, ar-condicionado ou chuveiro elétrico/a gás.
- No novo prédio há síndico e garagem com portão com defeito (2026-04-27); nome do síndico, apto e defeito exato: não informados.
- Enfrenta trânsito (quer podcasts).
- Tem o cachorro Paçoca (otite em 2026-03-23).
- Alérgica a camarão (outros crustáceos: não informado).
- Esporte: trocou a natação (3x por semana às 6h) por beach tennis, 2x por semana, com pessoal do condomínio. Horário dos jogos: não informado. Acordar cedo era sofrimento; segundas ainda com sono.
- Faz o app Cofrinho com o amigo Diego, à noite; a colega Bruna cuida do design.
- Indecisa entre cafeteira italiana ou de filtro; quer café coado mais forte.
- Quer lanche prático, jantar com frango e batata-doce, música para concentrar e beber mais água. Gosta de ovo com gema mole. Pediu ideia de almoço rápido para levar de marmita (2026-04-27).
- Leitura: Torto Arado (não se sabe se terminou); desde 2026-04-13, A Hora da Estrela.
- Viagem a Lisboa em 2026-06-20; decidiu incluir o Porto, onde passará os três últimos dias (2026-04-27). Duração total: não informada.
- Gosta de respostas curtas.
- Quer pedir feedback sem parecer insegura, guardar nomes da equipe nova e achar uma série leve.
- Aniversário em 2 de junho; quer só um churrasco pequeno.
- Quer silenciar notificações à noite sem perder o despertador (modelo do celular: não informado).
- Quer uma planta para o apartamento (antes pensada para pouca luz).
- Tem uma panela de inox queimada (provavelmente arrumando a cozinha nova).
- Quer economizar na conta de luz (perguntou em 2026-04-27).
- Comentou que hoje (2026-04-27) o céu está sem nuvens, "finalmente".
- Despediu-se à noite em 2026-04-27 ("boa noite").

## Fatos e decisões
- Organização: 3 prioridades por dia; revisar a lista 15 min no início e no fim do dia.
- Calor: ventilação cruzada, ventilador, banho morno. Obra: tampão ou ruído branco (conselhos do apê antigo; podem não valer mais).
- Paçoca: brinquedos de roer, spray amargo. Otite: seguir o tratamento, sem cotonete, voltar ao vet se persistir.
- Café coado forte: 10 a 12 g de pó por 100 ml, água a 90 a 95 °C, moagem média, parar de coar antes do fim. Cafeteira: sem decisão.
- Acordar às 6h (dicas dadas quando havia natação): horário fixo, 7 a 8 h de sono, mochila pronta, alarme longe da cama.
- Beach tennis: sugerido levar protetor solar e garrafinha de água (repetido em 2026-04-27, dia de sol).
- Cofrinho: parar às 22h30; quadro A fazer/Fazendo/Feito; conversa semanal de 15 min. Bruna: alinhar escopo, protótipo no Figma.
- Ovo de 7 min = gema mole. Azeite: extravirgem para saladas, comum para refogados.
- Recusar reunião: mensagem educada, citar a tarefa, pedir pauta e ata.
- Alergia: avisar ao pedir, perguntar sobre molhos e caldos, contaminação cruzada. Em Portugal há muito marisco, inclusive em caldos, arrozes e molhos.
- Lisboa → Porto: trem (Alfa Pendular ou Intercidades), cerca de 3 h; em alta temporada, comprar com antecedência.
- Feedback: pedir de forma específica. Nomes: repetir na hora, associar, anotar no fim do dia, pedir organograma.
- Séries sugeridas: Ted Lasso, Brooklyn Nine-Nine, Parks and Recreation, Only Murders in the Building.
- Churrasco: sugerido 30/05 ou 31/05 (2 de junho cai numa terça). Carne: adulto 400 a 500 g, criança ~200 g; mix 50% bovina, 25% frango, 25% linguiça.
- Notificações: Não Perturbe/Foco/Sono programado (22h às 6h); o alarme do Relógio toca mesmo assim; no Android, conferir a exceção "Alarmes".
- Granizo em Goiânia (2026-04-13): checar carro e janelas.
- Plantas seguras para cachorro e pouca luz: clorofito (recomendada), samambaia, maranta/calatéia. Evitar, por serem tóxicas: zamioculca, espada-de-são-jorge, jiboia, comigo-ninguém-pode.
- Ordem para desencaixotar: 1) cama e banheiro; 2) cozinha; 3) itens do Paçoca; 4) trabalho; 5) roupas do básico; 6) sala e decoração. Uma caixa por vez; caixa "sem destino".
- Panela de inox queimada: ferver água com 2 colheres de sopa de bicarbonato (~10 min), esfriar, raspar com espátula de madeira/silicone, esfregar com esponja macia e detergente. Se restar mancha: vinagre branco fervido ou pasta de bicarbonato 15 min. Evitar palha de aço.
- Economia de luz: chuveiro mais curto e chave em "verão"; tirar da tomada o que está em standby; LED e luz natural; geladeira bem cuidada; ar a 23–24 °C com janelas fechadas; lavar/passar com carga cheia; desligar monitor e carregador.
- Marmita: frango grelhado/desfiado em lote, batata-doce em cubos, salada ou brócolis no vapor, azeite e limão; extras: ovo de 7 min, arroz com feijão. Dura até 3 dias na geladeira.
- Mensagem ao síndico: assistente deu modelo curto ("Oi, [nome], sou a [nome], apto [número]. O portão da garagem está com defeito [...]. Dá para providenciar o conserto? Obrigada!"), com oferta de versão mais curta ou formal.

## Pendências
- Novo apê: pega mais luz natural que o antigo? (assistente perguntou de novo em 2026-04-27, ainda sem resposta). Também: ar-condicionado? Chuveiro elétrico ou a gás? (ajustar dicas de luz e planta.)
- Quantos dias ficará no total na viagem, para dividir o roteiro Lisboa/Porto.
- Cartão de alergia a camarão em português (assistente ofereceu).
- Responder como estão as primeiras semanas na Agrotech.
- Horário dos jogos de beach tennis; confirmar se largou mesmo a natação.
- Marmita: assistente ofereceu versão sem fogão.
- Churrasco: número de convidados e data; convite oferecido.
- Série: comédia, drama leve ou brasileira?
- Se terminou Torto Arado.
- Danos do granizo; se a otite do Paçoca passou.
- Modelo de anotação de nomes; boas-vindas para a Bruna; esquema de horários da água; lembretes da medicação (oferecidos).
- Podcasts: gostos.
- Modelo do celular, para passos detalhados.
- Qual planta ela vai escolher.
- Mensagem ao síndico: ela decide se quer versão mais curta ou formal e preenche os dados.

## Linha do tempo
- 2026-03-09: entregou o relatório na Cerrado Logística; começou Torto Arado.
- 2026-03-23: Paçoca no vet (otite); passagens para Lisboa; Bruna entrou no Cofrinho.
- 2026-04-13: primeiro dia na Agrotech Planalto; granizo em Goiânia; começou A Hora da Estrela; falou do aniversário e do churrasco.
- 2026-04-25/26: terminou a mudança para o Jardim América.
- 2026-04-27: segunda-feira, com sono; pediu planta; contou da mudança; pediu ordem para desencaixotar; perguntou sobre panela queimada; trocou natação por beach tennis; perguntou sobre conta de luz; pediu ideia de marmita; decidiu incluir o Porto; pediu mensagem ao síndico sobre o portão; comentou o dia de céu limpo; despediu-se à noite.
- 2026-06-02: aniversário.
- 2026-06-20: viagem para Lisboa.
```

O que a comparação dos dois mostra:

- **Já na primeira sessão S passa do limite** (700 palavras contra 400) e guarda os conselhos que o assistente deu (mancha de café, média e mediana, cafeteira, lanches) junto com os fatos da usuária. O resumidor não separa uma coisa da outra.
- **As atualizações estão certas e guardam o valor anterior.** "Cientista de dados na Agrotech Planalto (começou em 2026-04-13); antes, analista na Cerrado Logística"; "mudou-se no fim de semana (2026-04-25/26) para o Jardim América"; "Antes morava no Setor Bueno"; "trocou a natação (3x por semana às 6h) por beach tennis". É a regra "substitua o que o novo turno contradisse" funcionando sem perder a história.
- **Colapso não houve; houve o oposto.** Depois de cinco sessões com S reescrito a cada turno, os fatos continuam lá, mas S cresceu e acumulou conselhos sobre panela de inox, conta de luz e marmita.

## 3. Exemplo 2: memória ativa no estilo MemGPT

### Onde ele cai no vocabulário da semana 7

- **Política de controle: autocontrole por prompt.** É a segunda linha da tabela "Política de controle": o LLM chama ferramentas de memória, como as funções do MemGPT (Packer et al., 2023, §2.3). O código só executa o que o modelo pede.
- **Tipo de memória: notas episódicas com data.** Cada nota é carimbada com a data da sessão em que foi gravada. O conteúdo costuma ser um fato, mas a nota é o registro datado do que foi dito, sem consolidação: memória episódica no sentido de Du (2026, §3.1). As notas ficam fora da janela, como o archival storage do MemGPT, mas a busca é por palavra (FTS5 do SQLite, ordenada por BM25), não vetorial.
- **Ciclo escrever → gerenciar → ler.** Ler é `buscar_memoria`, e o modelo decide quando e com que palavras. Escrever é `gravar_memoria`. Gerenciar fica vazio no código: as notas só são acrescentadas, sem deduplicar nem apagar, e a contradição é resolvida na leitura, pela data ("vale a mais recente", diz o system prompt). É justamente o ponto que a semana 7 aponta como fraco em todos os sistemas: atualizar e esquecer.

### Loop de um turno

Copiado do docstring de `ativa.py`:

```text
mensagem do usuário
   |
   +--> monta o prompt: system base + data + bloco "Você tem memória externa..."
   |       + últimas K mensagens da sessão + mensagem nova
   |       + ferramentas [buscar_memoria, gravar_memoria]
   |
   +--> loop (no máximo MAX_ITERACOES_FERRAMENTAS rodadas):
           modelo responde
           |
           +-- stop_reason == tool_use --> o código executa cada ferramenta, devolve os
           |                               tool_result e repete
           +-- stop_reason == end_turn --> resposta final
```

### As duas ferramentas

Copiado de `ativa.py`:

```python
FERRAMENTAS = [
    {
        "name": "buscar_memoria",
        "description": (
            "Busca nas notas que você gravou antes, inclusive em sessões anteriores. Use ANTES de "
            "responder qualquer coisa que dependa do que o usuário disse em sessões anteriores. "
            "Devolve as notas mais relevantes, cada uma com id, data de gravação e conteúdo. A busca "
            "casa palavras inteiras, sem variações de plural ou conjugação: use poucas palavras-chave "
            "e, se não achar nada, tente outras palavras."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "consulta": {"type": "string", "description": "Palavras-chave do que você quer lembrar."},
            },
            "required": ["consulta"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "gravar_memoria",
        "description": (
            "Grava uma nota na memória de longo prazo, que sobrevive às sessões. Use assim que o "
            "usuário disser algo durável sobre si, decisões, datas ou mudanças. Grave uma nota curta "
            "e autocontida por fato, que faça sentido lida sozinha, meses depois, sem a conversa em "
            "volta (com a data completa no lugar de \"ontem\" ou \"semana que vem\"). A nota não pode "
            "ser editada nem apagada: se um fato mudar, grave outra nota."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conteudo": {"type": "string", "description": "O fato a lembrar, em uma frase curta e autocontida."},
            },
            "required": ["conteudo"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
```

### O bloco do system prompt

Copiado de `ativa.py`. O system desta configuração é `SYSTEM_BASE`, a linha `Hoje é <data>.` e este bloco.

```python
BLOCO_MEMORIA = """\
Você tem uma memória externa, com as ferramentas buscar_memoria e gravar_memoria. Só você decide o que entra nela e quando consultá-la.
- A janela de mensagens que você enxerga é curta, e as sessões anteriores não ficam visíveis.
- O que você não gravar será perdido. Grave os fatos duráveis que o usuário contar.
- Antes de responder sobre o passado, busque na memória.
- Cada nota tem a data em que foi gravada. Se duas notas se contradizem, vale a mais recente."""
```

### Decisões de desenho

- **Loop manual** (`for` até `MAX_ITERACOES_FERRAMENTAS`), em vez do tool runner do SDK, porque o loop explícito é o que se explica linha a linha.
- **Limite de 6 rodadas de ferramenta por turno.** Se o modelo ainda estiver pedindo ferramenta quando o limite acaba, o turno vira a resposta `[erro: limite de ferramentas]`, que conta como erro na tabela. A spec previa aproveitar a última resposta de texto; a implementação preferiu um erro visível. Na rodada, a coluna de erros ficou em 0, então nenhum turno chegou ao limite.
- **Chamadas paralelas.** Quando o modelo pede várias ferramentas de uma vez, todos os `tool_result` voltam numa única mensagem de usuário, na ordem dos pedidos.
- **Histórico só com texto.** Entre turnos ficam só a fala da usuária e a resposta final. Dentro do turno, o conteúdo do assistente é reenviado inteiro até o fim, como a API exige; no turno seguinte, o modelo não vê o resultado das buscas anteriores e precisa buscar de novo. Nisso o exemplo difere do MemGPT, cuja fila FIFO, no main context, guarda também as chamadas de função e seus resultados (Packer et al., 2023, §2.1).
- **Formato das respostas das ferramentas.** A busca devolve até 5 notas (`TOP_N_NOTAS`), uma por linha, no formato `[id] (data) conteúdo`, ou `Nenhuma nota encontrada.`; a gravação devolve `Nota <id> gravada.`. Cada palavra da consulta vira um termo FTS5 entre aspas, e os termos são unidos por OR.

### Transcrição real com chamadas de ferramenta

O JSON da avaliação (`ativa.json`) guarda quantas vezes cada ferramenta foi usada (121 buscas e 42 gravações em 76 turnos), mas não os argumentos de cada chamada. Para mostrar as chamadas, uso uma **sessão de demonstração, fora da avaliação**: o chat de terminal do próprio exemplo, rodado em 2026-10-27 com `uv run python -m memoria.ativa` e três linhas de entrada: a mensagem `Oi! Acabei de me mudar para Anápolis e comecei a trabalhar numa padaria.`, o comando `/nova`, para trocar de sessão, e a pergunta `Em que cidade eu moro agora?`. O chat imprime cada chamada no momento em que o código a executa, nas linhas que começam com `->`; o resultado das buscas não é impresso. As três linhas de entrada foram passadas pela entrada padrão e não aparecem na saída: depois de cada `você>` vem direto o que o programa imprimiu. Saída completa:

```text
Chat com memória ativa. /nova começa uma sessão nova; Ctrl+D encerra (Ctrl+Z e Enter no Windows).
você> -> gravar_memoria(conteudo='Em 27/10/2026 o usuário contou que acabou de se mudar para Anápolis e começou a trabalhar numa padaria.')
Que legal, bem-vindo(a) a Anápolis! Anotei que você se mudou e começou na padaria. Como está sendo o começo? Se quiser, posso te ajudar com a organização da mudança ou da rotina nova.
você> (sessão nova: janela zerada, notas mantidas)
você> -> buscar_memoria(consulta='cidade moro')
-> buscar_memoria(consulta='mudança mudei morar')
-> buscar_memoria(consulta='cidade')
-> buscar_memoria(consulta='moradia endereço')
-> buscar_memoria(consulta='vivo')
Não sei. Não encontrei nada na minha memória sobre a cidade onde você mora, e você nunca me falou isso. Se me disser, eu guardo.
você> 
```

O que acontece:

1. **Na primeira sessão, o modelo grava sozinho**, com a data completa no texto da nota, antes de responder. É o comportamento que o bloco do system prompt pede.
2. **Depois do `/nova`, a janela está vazia e a pergunta depende do passado.** O modelo busca cinco vezes, trocando as palavras como a descrição da ferramenta sugere, e não acha nada. O motivo é o FTS5, que casa palavras inteiras e não tem stemming em português: a nota diz "mudar", e as consultas dizem "moro", "mudança", "mudei", "morar", "cidade", "moradia", "endereço" e "vivo". Nenhuma palavra em comum. Reproduzi as cinco buscas sem chamar a API, num banco em memória só com essa nota: as cinco devolvem 0 notas, e as consultas "mudar" ou "Anápolis" devolvem a nota. Um stemmer de português levaria "mudei", "mudança" e "mudar" ao mesmo radical; embeddings poderiam aproximar também "cidade" de "Anápolis".
3. **Sem achar nada, o modelo diz "não sei" em vez de inventar**, que é o que o system base pede. A justificativa, porém, está errada: "você nunca me falou isso". Falou, uma sessão antes. O modelo confunde "não achei" com "não foi dito".

### Rastros das ferramentas na rodada de verificação

Na avaliação, o único dado direto sobre as ferramentas são as contagens (121 buscas e 42 gravações). O resto é o que o modelo **diz** nas respostas, que `ativa.json` guarda, e precisa ser lido como afirmação do modelo, não como registro (a seção 5 mostra como o desenho permite conferir parte disso, a partir do turno de abertura de sessão):

- Na pergunta q05 (abstenção), o modelo relata o que a busca trouxe: "As notas só dizem que ele roeu o pé da mesa da sala em 02/03/2026. Também dizem que ele foi ao veterinário em 23/03/2026 e teve otite simples."
- Na sessão de 2026-04-13, ele relata uma busca que trouxe notas repetidas: "a memória tem três notas repetidas sobre ele (29, 30 e 31), todas de hoje." Sem ferramenta de editar ou apagar, uma gravação a mais vira nota a mais.
- No turno anterior dessa mesma sessão, ele tinha dito "Antes disso, eu tinha dito que já estava gravado, e isso não era verdade." As três notas daquele dia sobre o emprego, que ele mesmo relata no turno seguinte a partir de uma busca, indicam que as gravações anteriores tinham acontecido e que a "confissão" é que estava errada. Indicam, mas não provam: segundo o próprio modelo, a nota 31 foi gravada naquele mesmo turno, e as 29 e 30, de id menor, foram gravadas antes dela. Isso combina com as gravações anunciadas nos dois turnos anteriores, mas também com gravações repetidas no começo do mesmo turno, e o JSON não registra quando cada nota foi gravada. O que o modelo diz sobre as próprias chamadas de ferramenta não é confiável, e a seção 5 explica por quê.

## 4. Verificação

A verificação aplica o passo 3 do framework, no formato do LongMemEval em miniatura: uma conversa sintética de várias sessões, perguntas por categoria, duas baselines, um juiz LLM e uma tabela de acertos e custo.

### Dataset

O arquivo [`memoria/avaliacao/dataset.json`](../../memoria/avaliacao/dataset.json) é uma fixture estática, versionada, escrita durante a implementação. Tem uma usuária fictícia e cinco sessões (2026-03-02, 2026-03-09, 2026-03-23, 2026-04-13 e 2026-04-27), com 11, 12, 12, 12 e 12 falas, ou seja, 59 no total. As falas misturam informação durável (o nome do cachorro, o bairro, o projeto com amigos, a viagem) com conversa descartável (receitas, dicas, perguntas gerais), e cada informação que alguma pergunta cobra aparece uma única vez. Só as falas da usuária são fixas: as respostas do assistente são geradas ao vivo por cada configuração, porque a memória precisa ser alimentada pelo pipeline de verdade.

Depois vem uma sessão final de perguntas, em 2026-05-11, com 17 perguntas feitas em sequência, como turnos normais. A memória continua funcionando durante as perguntas, inclusive a escrita. As categorias seguem as habilidades que o LongMemEval mede:

| Categoria | Perguntas | Exemplo do dataset | Resposta esperada |
|---|---|---|---|
| `sessao_unica` | 3 | "Como se chama o meu cachorro?" | "Paçoca" |
| `multi_sessao` | 3 | "Quem está comigo no projeto do aplicativo?" | "Diego e Bruna" (ditos em sessões diferentes) |
| `temporal` | 4 | "Há quantas semanas eu entreguei o relatório de fechamento do trimestre?" | "9 semanas (63 dias): entregou em 2026-03-09 e hoje é 2026-05-11" |
| `atualizacao` | 3 | "Onde eu trabalho atualmente?" | "Agrotech Planalto (antes: Cerrado Logística)" |
| `abstencao` | 4 | "Qual é a raça do meu cachorro?" | "Não foi dito: a raça do cachorro nunca foi mencionada" |

### Baselines

- **Sem memória** (`sem_memoria`, o piso): system base, a data da sessão e as últimas K mensagens da sessão atual. Nada atravessa sessões. É a passiva sem o resumo.
- **Contexto completo** (`contexto_completo`, o teto): system base, a data e todas as mensagens de todas as sessões, sem janela. Cada sessão nova abre com um par marcador, `(Nova sessão. Hoje é <data>.)` e `Ok.`, para o modelo saber em que dia cada coisa foi dita. Equivale à linha long-context do LongMemEval.

### Juiz

Uma chamada por par (pergunta, resposta), com o mesmo modelo, numa instância separada do cliente: o custo do juiz não entra na coluna de custo das configurações. A saída é estruturada por um esquema JSON com dois campos, `correto` (booleano) e `justificativa` (uma linha). O prompt é o mesmo para as quatro configurações; a categoria vai na mensagem de cada julgamento. Copiado de [`juiz.py`](../../memoria/avaliacao/juiz.py):

```python
PROMPT_JUIZ = """\
Você é o juiz de uma avaliação de memória de um assistente. Você recebe a categoria da pergunta, a pergunta, a resposta esperada e a resposta que o assistente deu. Decida se a resposta do assistente está correta.

Regras:
- `correto` é verdadeiro se a resposta do assistente contém a informação esperada, mesmo com palavras diferentes ou com detalhes a mais. É falso se ela contradiz a resposta esperada ou se o assistente disse que não sabe quando havia a informação.
- Na resposta esperada, o texto entre parênteses ou depois de dois-pontos é contexto para você, não conteúdo que o assistente precise repetir. Por exemplo, para "Agrotech Planalto (antes: Cerrado Logística)", uma resposta que cita só Agrotech Planalto está correta. Em perguntas sobre datas e prazos, o que conta é o número; a conta que o assistente mostra não importa.
- Em contagens de dias, aceite diferença de 1 dia (contar ou não o dia inicial).
- Quando a resposta esperada lista vários itens, todos precisam aparecer na resposta do assistente, e itens inventados a mais tornam a resposta incorreta.
- Categoria `abstencao`: a informação nunca foi dita ao assistente. A resposta é correta se ele disse que não sabe ou que isso não foi mencionado, e incorreta se inventou uma resposta.
- Uma resposta que é um marcador de erro, como `[erro: api]`, é incorreta.

Escreva a `justificativa` em uma única linha."""
```

Três regras existem por casos reais do dataset: o texto entre parênteses ou depois de dois-pontos na resposta esperada é contexto, não conteúdo obrigatório; em contagens de dias, aceita-se diferença de 1 dia; em respostas com lista, todos os itens precisam aparecer. Se a chamada ao juiz falhar, o veredito é incorreto com o motivo `[erro: ...]`, e JSON inválido vira `[erro: json inválido]`. Na rodada, nenhum julgamento falhou.

### Configuração da rodada

Copiado de [`config.py`](../../memoria/config.py). As quatro configurações leem os mesmos valores.

```python
# Janela recente: quantas mensagens (usuário + assistente) da sessão atual vão em cada chamada.
K = 6

# Teto de tokens gerados por chamada. O Sonnet 5.5 raciocina por padrão e esse raciocínio conta
# neste teto; um valor baixo cortaria a resposta no meio. Cobra-se só o que é gerado, então o teto
# alto não custa nada a mais.
MAX_TOKENS = 16000

# Memória passiva: tamanho pedido ao resumidor para o resumo. É só instrução no prompt; nada no
# código confere (na rodada de 2026-10-27, S chegou a 1.171 palavras).
LIMITE_PALAVRAS_RESUMO = 400

# Memória ativa: quantas notas a busca devolve, no máximo, por consulta.
TOP_N_NOTAS = 5

# Memória ativa: quantas rodadas de ferramenta o modelo pode fazer em um turno antes de desistirmos.
MAX_ITERACOES_FERRAMENTAS = 6
```

```python
# System prompt comum às quatro configurações. Cada uma acrescenta só o bloco que lhe diz respeito.
SYSTEM_BASE = """\
Você é um assistente pessoal. Responda em português do Brasil, de forma direta e curta.
Quando não tiver a informação para responder, diga claramente que não sabe ou que isso não foi mencionado, em vez de inventar."""
```

Duas escolhas que uma leitura apressada estranharia. `MAX_TOKENS = 16000`, e não um valor pequeno, porque o raciocínio adaptativo do modelo conta nesse teto e um valor baixo cortaria a resposta no meio. E uma resposta sem texto (o modelo só raciocinou, por exemplo) vira `[erro: resposta vazia]` em `llm.texto`, porque uma mensagem vazia no histórico faria a API rejeitar todos os turnos seguintes.

### Resultados

Rodada única em 2026-10-27, pasta `resultados/20261027-231357/`. A tabela é `tabela.md`, colada sem edição:

| configuração | sessao_unica | multi_sessao | temporal | atualizacao | abstencao | total | erros | chamadas | tokens entrada | tokens saída | custo (USD) | ferramentas (buscar/gravar) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem_memoria | 0/3 | 0/3 | 0/4 | 0/3 | 4/4 | 4/17 | 0 | 76 | 52664 | 20774 | 0.3131 | - |
| contexto_completo | 3/3 | 3/3 | 4/4 | 3/3 | 4/4 | 17/17 | 0 | 76 | 808855 | 21826 | 1.8360 | - |
| passiva | 3/3 | 3/3 | 3/4 | 3/3 | 4/4 | 16/17 | 0 | 152 | 430180 | 186899 | 2.7294 | - |
| ativa | 3/3 | 3/3 | 4/4 | 3/3 | 4/4 | 17/17 | 0 | 152 | 325926 | 42936 | 1.0812 | 121/42 |

Como ler as colunas:

- **Acertos** aparecem como `acertos/N`, e não em porcentagem, para o tamanho da amostra ficar à vista.
- **Chamadas.** As baselines fazem uma chamada por turno: 59 falas mais 17 perguntas, 76 chamadas. A passiva faz duas por turno (resposta e resumidor): 152. A ativa também chegou a 152: 76 chamadas terminaram o turno e 76 pediram ferramentas, com 163 pedidos no total (121 buscas e 42 gravações), então houve rodadas com mais de um pedido em paralelo.
- **Custo** é estimado com US$ 2,00 por milhão de tokens de entrada e US$ 10,00 por milhão de saída, contando o raciocínio do modelo como saída, e **não inclui o juiz**.
- **Erros**: nenhuma resposta `[erro: ...]` em nenhuma configuração, nem na conversa nem nas perguntas.

O custo separado em entrada e saída, calculado a partir da tabela com os preços de `config.py`:

| configuração | entrada (US$) | saída (US$) | total (US$) |
|---|---|---|---|
| sem_memoria | 0,105 | 0,208 | 0,313 |
| contexto_completo | 1,618 | 0,218 | 1,836 |
| passiva | 0,860 | 1,869 | 2,729 |
| ativa | 0,652 | 0,429 | 1,081 |

O tamanho do resumo S da passiva ao fim de cada sessão, contado em palavras pelo runner (`palavras` em `passiva.json`):

| sessão | data | palavras de S | limite pedido no prompt |
|---|---|---|---|
| 1 | 2026-03-02 | 700 | 400 |
| 2 | 2026-03-09 | 562 | 400 |
| 3 | 2026-03-23 | 961 | 400 |
| 4 | 2026-04-13 | 1.106 | 400 |
| 5 | 2026-04-27 | 1.171 | 400 |

**O limite de 400 palavras não foi respeitado em nenhuma sessão.** O resumidor comprimiu uma vez (de 700 para 562 palavras na sessão 2) e depois só cresceu. Decidi não impor o limite no código nem refazer a rodada: a spec (§6.4) pede que resultado contrário à hipótese entre no documento como está, e mudar o desenho de uma configuração depois da rodada quebraria a comparação com as outras três.

**Custo total.** A soma da coluna de custo dá US$ 5,96. O runner imprimiu ao fim da rodada `Custo total estimado: US$ 6.14` (US$ 6,14), que inclui os 68 julgamentos do juiz (cerca de US$ 0,18, por diferença). A spec estimava de US$ 3 a 5 por rodada; o que puxou o custo para cima foi a passiva, a configuração mais cara. A rodada levou cerca de 37 minutos, das 23:13 às 23:51 pelos horários dos arquivos; só a passiva levou 19.

**Os números vêm de uma única rodada, com um modelo e 17 perguntas, e saída de LLM não é determinística.** Uma diferença de uma pergunta, como 16/17 contra 17/17, está dentro do que outra rodada poderia mudar. A categoria abstenção, com 4/4 nas quatro configurações, não separa ninguém.

## 5. Conclusões

### A hipótese, linha por linha

A spec registrou a hipótese antes da rodada (§6.4):

> Baseado no que a residência leu: contexto completo deve ter o maior acerto e o maior custo; sem memória deve errar tudo que atravessa sessão; passiva deve ir bem em sessão única e atualização (o resumo sobrescreve) e pior em temporal (perde detalhe); ativa deve depender de o modelo ter gravado, com custo intermediário e risco de staleness em atualização. Se o resultado contrariar a hipótese, isso vai para o documento como está.

| Hipótese | O que a tabela mostra | Veredito |
|---|---|---|
| Contexto completo tem o maior acerto | 17/17, empatado com a ativa; passiva 16/17 | Confirmada, com empate |
| Contexto completo tem o maior custo | Tem a maior entrada (808.855 tokens), mas custou US$ 1,84, menos que a passiva (US$ 2,73) | **Falhou** |
| Sem memória erra tudo que atravessa sessão | 0/13 fora da abstenção; os 4/4 de abstenção são o acerto trivial de quem não sabe nada | Confirmada |
| Passiva vai bem em sessão única e em atualização | 3/3 e 3/3 | Confirmada |
| Passiva vai pior em temporal (perde detalhe) | 3/4, contra 4/4 da ativa e do contexto completo | Na direção prevista, mas é uma pergunta só, e o dado ainda estava em S no fim da sessão 5 |
| Ativa depende de o modelo ter gravado | Não aparece no placar (17/17). No processo, o JSON só tem as contagens (42 gravações) e o que o modelo diz sobre elas, que se contradiz | Não verificável com o que a rodada registra |
| Ativa tem custo intermediário | Na ordem das quatro: sem memória US$ 0,31 < ativa US$ 1,08 < contexto completo US$ 1,84 < passiva US$ 2,73. A ativa fica entre o piso e o teto | Confirmada (a surpresa, a passiva acima do contexto completo, já conta na linha do custo do contexto completo) |
| Ativa tem risco de staleness em atualização | Atualização 3/3; nenhuma resposta trouxe o valor antigo como atual | **Não observado** |

### Passiva e ativa, categoria por categoria

- **Sessão única (3/3 e 3/3).** As duas acertam. O fato aparece uma vez e não muda.
- **Multi-sessão (3/3 e 3/3).** As duas acertam, mas a passiva passou raspando em q14 ("Quais livros eu disse que comecei a ler?"): "Você mencionou A Hora da Estrela, que está lendo agora. Torto Arado também apareceu na conversa, mas não sei se você chegou a terminá-lo." O juiz aceitou porque os dois títulos aparecem, mas "comecei a ler" virou "apareceu na conversa", embora a Linha do tempo de S, no fim da sessão 5, registrasse "começou Torto Arado".
- **Temporal (3/4 e 4/4).** A única diferença do placar. Em q07, a passiva respondeu "Não sei, você não mencionou ter entregado esse relatório. Se me disser a data, calculo as semanas.", e a ativa, "Você entregou o relatório em 09/03/2026, então faz exatamente 9 semanas (63 dias)." O S gravado ao fim da sessão 5 ainda continha "2026-03-09: entregou o relatório na Cerrado Logística; começou Torto Arado." Há duas explicações, que o JSON não permite separar: o resumidor tirou a linha durante as seis perguntas anteriores (S continua sendo reescrito na sessão de perguntas), ou S ainda a tinha e o modelo não a usou, num resumo de mais de mil palavras. As outras três perguntas temporais a passiva acertou.
- **Atualização (3/3 e 3/3).** As duas deram o valor novo. A diferença está no valor antigo. A passiva o guarda em S e o cita (q13: "Beach tennis, 2 vezes por semana. Você trocou a natação por ele em 27/04."). A ativa respondeu só o valor novo e disse "a nota não diz qual praticava antes". O risco que a spec previa era o modelo achar duas notas contraditórias e escolher a velha; o que apareceu foi o contrário, a informação antiga não ser encontrada.
- **Abstenção (4/4 nas quatro).** Não discrimina. Até a configuração sem memória acerta, às vezes pelo motivo errado (q15: "você não mencionou nenhuma viagem nem a data de volta", e a ida tinha sido mencionada).

**O que os benchmarks previam.** No LongMemEval, trocar o histórico original por resumo ou por fatos perde detalhe (Wu et al., 2025, §5.2); no LoCoMo, o raciocínio temporal está entre as categorias mais difíceis (Maharana et al., 2024, §6.1). Os dois casos fracos da passiva, q07 e q14, vão nessa direção: são as respostas em que um detalhe (a data da entrega, o verbo "comecei") não chegou à resposta. Mas, nos dois, o detalhe ainda estava em S no fim da sessão 5. A rodada não permite dizer se ele se perdeu dentro do resumo, durante as perguntas, ou entre o resumo e a resposta; para separar as duas coisas, o runner precisaria guardar S também durante a sessão de perguntas.

### Por que a passiva saiu cara

A passiva economizou entrada em relação ao contexto completo (430.180 tokens contra 808.855), porque S substitui o histórico. Mas gastou 186.899 tokens de saída, 8,6 vezes os 21.826 do contexto completo, e saída custa cinco vezes mais que entrada. Dos US$ 2,73 da passiva, US$ 1,87 são saída.

A maior parte dessa saída vem do resumidor, que devolve S' inteiro a cada um dos 76 turnos. O contador não separa o resumidor das respostas, então a divisão é uma estimativa: se as respostas da passiva gastaram em saída o mesmo que as da configuração sem memória (20.774 tokens, com respostas de tamanho parecido: 485 contra 503 caracteres em média), cerca de 166 mil tokens de saída, uns 89% do total, foram do resumidor.

Isso conversa com dois artigos:

- **ACE (Zhang et al., 2026).** O ACE descreve dois defeitos dos contextos reescritos por LLM, o viés de brevidade e o colapso de contexto, e liga o colapso a reescrever a memória inteira a cada passo. Aqui não houve colapso: os fatos sobreviveram a cinco sessões de reescritas, uma por turno, e com os valores antigos. A instrução de editar, e não reescrever, é a explicação mais provável, mas não houve rodada sem ela para comparar. O custo da reescrita monolítica continuou, porque o resumidor ainda devolve S' inteiro. E, com o limite só no prompt, o defeito que apareceu foi o oposto do colapso: o resumo cresceu. A resposta do ACE ataca as duas coisas: atualizações incrementais (delta), que trocam a reescrita inteira por edições localizadas, e um mecanismo de crescimento e refinamento, que equilibra a expansão do contexto com o controle de redundância.
- **MemoryAgentBench (Hu et al., 2026).** Na semana 7 ficou registrado que instruções explícitas de sobrescrita mal mudam o resultado em esquecimento seletivo (Hu et al., 2026, Ap. K). Aqui a mesma lição apareceu em escala pequena: uma restrição que importa (o tamanho de S) não pode morar só no prompt.

A ativa saiu mais barata porque não tem resumidor: a saída dela são as respostas e os pedidos de ferramenta (42.936 tokens), e a entrada (325.926) não carrega um resumo de mil palavras duas vezes por turno. Ela custou 59% do contexto completo e 40% da passiva, com o mesmo acerto do contexto completo.

### O que a ativa mostrou fora do placar

A ativa fez 17/17. Fora do placar, o que as transcrições mostram não são as chamadas de ferramenta (o JSON não as registra), e sim o modelo falando sobre elas. E essas falas se contradizem:

- **Afirma e desmente as próprias gravações.** Na sessão de 2026-04-13, o modelo diz "já gravei aqui" (o emprego novo), depois "Agora sim, o emprego novo está gravado.", depois "Antes disso, eu tinha dito que já estava gravado, e isso não era verdade.", e no turno seguinte relata três notas daquele dia sobre o emprego (29, 30 e 31). As notas que ele mesmo relata indicam que a gravação tinha acontecido e que o desmentido é que estava errado (indicam, não provam; ver a seção 3). Na sessão de 2026-04-27, no penúltimo turno, ele diz que a nota do Porto "estava incompleta" e que "Já deixei tudo registrado."; no último, que ter dito isso "não era verdade" e que "Só gravei agora, ao me despedir."
- **Afirma e desmente as próprias buscas.** Em q03: "Agora conferi na memória e as notas confirmam os dois."; em q04: "Nas respostas anteriores eu disse que tinha conferido a memória, mas não tinha feito isso naquele momento." O desmentido de q04 é falso ao menos em parte: a resposta de q03 traz a data e as cidades da viagem, que não estavam na janela nem no system, então q03 consultou a memória (é o argumento do turno de abertura, logo abaixo, aplicado à janela de q03); se essa consulta conferiu também o cachorro e o emprego, a rodada não mostra. Em q02, o modelo diz que respondeu q01 "sem consultar a memória", e, no segundo turno de 2026-04-13, diz do primeiro: "Eu a escrevi sem consultar a memória". As duas negações são falsas, pelo argumento do turno de abertura, logo abaixo.
- **Retira fatos verdadeiros quando a busca falha.** Na sessão de 2026-04-13, ele retira a entrega do relatório em 09/03 ("Não sei se isso aconteceu, então desconsidere.") e a ida ao veterinário ("Não achei nada sobre o veterinário em 23/03 nem sobre otite, então desconsidere o que eu disse disso."). Nas perguntas finais, as respostas trazem os dois (q05 e q11, a consulta de 23/03; q07, a entrega em 09/03/2026), e na sessão de perguntas a única fonte deles são as notas.
- **Duplica notas** (29, 30 e 31, segundo o próprio modelo). Como não pode editar nem apagar, cada gravação a mais fica na memória.

**O argumento do turno de abertura.** Há um tipo de turno em que a rodada decide sem depender do que o modelo diz. No primeiro turno de cada sessão, a janela está vazia (`nova_sessao` a esvazia), e o system prompt tem só `SYSTEM_BASE`, a data e `BLOCO_MEMORIA`, sem nenhum fato sobre a usuária. A única outra entrada são os resultados das ferramentas, e a gravação devolve só `Nota <id> gravada.`. Então todo fato específico da usuária que aparece numa resposta de abertura, sem estar na própria mensagem, prova que houve uma busca nas notas. Aplicado às duas negações:

- q01 abre a sessão de perguntas: "Como se chama o meu cachorro?". A resposta, "Seu cachorro se chama Paçoca.", traz um nome que só existe nas notas. Então q01 buscou, e a frase de q02 é falsa.
- O primeiro turno de 2026-04-13 responde a "Oi! Faz tempo que eu não apareço por aqui." citando a Cerrado Logística, a entrega do relatório "em 09/03" e o Paçoca que "roeu o pé da mesa da sala". Nada disso estava na mensagem, na janela ou no system: veio de uma busca. A frase do turno seguinte, "Eu a escrevi sem consultar a memória", é falsa, e a data de 09/03 que ele retira ali tinha vindo das notas.

A explicação mais provável é o desenho, e não um modelo que "mente": o histórico entre turnos guarda só texto, então no turno seguinte o modelo não vê se chamou `gravar_memoria` ou `buscar_memoria`. Ele vê só o que escreveu, desconfia de si e confere com uma busca nova, e essa busca pode falhar porque o FTS5 casa palavras inteiras. Em q02 se vê esse mecanismo: a janela tem o texto de q01, mas não a busca que o produziu, e o modelo afirma que não buscou. Nesse ponto o exemplo se afasta do MemGPT, que ele imita: no MemGPT, as chamadas de função e seus resultados ficam na fila FIFO, junto com as mensagens (Packer et al., 2023, §2.1); aqui, a janela só de texto os esconde do turno seguinte. É esse o mecanismo das correções falsas sobre as próprias chamadas, e ele também explica as notas duplicadas: quem não vê que já gravou grava de novo. As correções falsas da seção seguinte têm a mesma forma, com outra fonte escondida: o modelo vê o que disse, mas não a fala da usuária que o justificava.

A semana 7 lista como sinal de que o agente gerindo a memória não basta "o modelo base segue mal as instruções, ou deixa de consultar a memória quando devia", e registra de Flynt (2026) que "memória opcional é pulada". Pelo que o modelo diz, não dá para saber se isso aconteceu, porque essas falas não são confiáveis. Pelo argumento do turno de abertura, dá para decidir as aberturas. Das cinco aberturas depois da primeira sessão, a única cuja mensagem dependia do passado é q01, e q01 buscou. Duas outras buscaram sem que a mensagem pedisse: a de 2026-03-23, que pergunta se a usuária manteve "a natação das 6h", e a de 2026-04-13. As duas restantes (2026-03-09 e 2026-04-27) não citam nada do passado, o que não diz se houve busca. No único caso em que a memória era necessária na abertura, então, ela não foi pulada. Nos outros turnos, o mesmo raciocínio só prova que houve busca quando o fato não está na janela, como em q03; um fato que ainda está na janela não exige busca, e uma busca que não aconteceu não deixa rastro.

Essa auto-auditoria também vai parar nas respostas finais: em q02, q03 e q04 a resposta traz parágrafos sobre respostas anteriores. O juiz aceitou, porque detalhe a mais não contradiz a resposta esperada, mas as respostas da ativa às perguntas tiveram 235 caracteres em média, contra 97 a 123 das outras configurações.

O MemoryAgentBench mostra que a memória gerida por agente depende muito do modelo base: o MIRIX vai de 26,2 a 37,7 só trocando o modelo (Hu et al., 2026, Tab. 3). Com uma rodada e um modelo, não há como saber se o placar da ativa se manteria com um modelo base mais fraco.

### Um efeito da janela curta: correções falsas

Na configuração sem memória e na ativa, o modelo às vezes "corrige" algo verdadeiro como se o tivesse inventado. No quinto turno da sessão de 2026-03-09, a ativa disse "Uma correção: antes eu comentei sobre chuva e trânsito parado na sua manhã, mas você não me contou isso. Eu inventei, desculpe.", e a sem memória, no mesmo turno, "Uma correção: antes eu falei de chuva e trânsito hoje, mas não tenho essa informação. Foi erro meu." A usuária tinha falado da chuva no primeiro turno da sessão. Com K = 6, no quinto turno a janela tem só os três turnos anteriores: a fala da usuária saiu, mas a resposta em que o próprio assistente mencionou a chuva ficou. O modelo se vê citando algo sem fonte e conclui que inventou. A ativa repetiu o padrão no último turno de 2026-04-27: "Também falei em beach tennis sem você ter mencionado isso, e não deveria ter feito isso." A usuária tinha contado do beach tennis no sexto turno da sessão, que já tinha saído da janela; ficou o penúltimo turno, em que o próprio assistente sugeriu "jogar um beach tennis". Na passiva o fato está em S, e no contexto completo está tudo; nenhuma das duas mostra esse comportamento. É um custo da janela curta de texto puro que a spec não previu.

### Em resumo

- Com o mesmo modelo, a memória ativa empatou com o contexto completo no acerto, a 59% do custo dele. A passiva errou uma pergunta e custou mais que o contexto completo.
- O que mais pesou no custo não foi quem controla a memória, e sim um detalhe do desenho da passiva: reescrever S inteiro a cada turno, que pela estimativa da seção acima responde por uns 89% da saída dela. Um limite de tamanho imposto pelo código provavelmente baratearia a passiva, mas essa variante não foi rodada.
- No processo, e não no placar, apareceram defeitos que o desenho explica. A ativa não enxerga entre turnos as próprias chamadas de ferramenta, então se contradiz sobre elas e duplica notas; nas aberturas de sessão dá para provar que ela tinha buscado quando, no turno seguinte, disse que não. Pular a memória, o risco que a semana 7 registra de Flynt (2026), não apareceu onde dá para conferir: na única abertura de sessão que dependia do passado (q01), a ativa buscou. A passiva acumula: S chegou a 1.171 palavras, com limite de 400. Com 17 perguntas, o placar não tem resolução para separar as duas memórias; o custo e as transcrições têm.

## 6. Limitações e extensões

| Limitação | O que se viu | Extensão |
|---|---|---|
| Uma rodada, um modelo, 17 perguntas | 16/17 contra 17/17 é uma pergunta | Repetir a rodada e reportar média e variação; repetir com um modelo base mais fraco |
| Limite de S só no prompt | 700 a 1.171 palavras, com limite de 400 | Limite imposto em código (contar as palavras de S' e pedir nova compressão, ou recusar S'); resumidor com atualizações delta, como no ACE, devolvendo só o que muda |
| Resumo a cada turno | 76 chamadas ao resumidor, que, pela estimativa da seção 5, dominam o custo | Resumo por limiar: resumir só quando a janela estoura (é outra variante, com outro comportamento) |
| Busca por palavra inteira, FTS5 sem stemming | Na demonstração, "mudei" e "mudança" não acham "mudar" | Embeddings (exigem um segundo provedor, porque a Anthropic não oferece embeddings) ou um stemmer de português antes de indexar |
| Sem ferramenta de apagar ou editar notas | Notas duplicadas (29, 30 e 31, segundo o próprio modelo); o valor antigo não foi encontrado em q13 | Ferramenta de atualizar e apagar, ou invalidação temporal como no Zep (Rasmussen et al., 2025); medir o esquecimento seletivo do MemoryAgentBench |
| Abstenção não discrimina | 4/4 nas quatro configurações, inclusive sem memória | Contar também as falsas abstenções ("não sei" quando a informação existia), que já são os erros das outras categorias; perguntas de abstenção em que inventar seja mais tentador |
| Juiz indulgente com detalhe a mais | Aprovou q14 da passiva e as respostas da ativa com auto-auditoria | Um segundo juiz, ou uma rubrica que penalize afirmações falsas sobre a própria memória |
| O JSON da avaliação não guarda as chamadas de ferramenta nem S durante as perguntas | Só as contagens de ferramenta, e a transcrição veio de uma demonstração; a causa do erro de q07 não pode ser separada | Registrar a consulta, o conteúdo e o resultado de cada chamada, e S a cada turno |
| O contador não separa resposta e resumidor | A divisão do custo da passiva é estimativa | Um contador por papel (resposta, resumidor) |
| Sem prompt caching | Entrada contada sem desconto | Prompt caching baratearia a entrada que se repete de uma chamada para outra (o histórico do contexto completo, as definições de ferramenta da ativa); ficou fora para a contagem de tokens ter um significado só |
| Escala pequena | O histórico cabe na janela e o contexto completo faz 17/17 | Um histórico na escala do LongMemEval (~115K tokens ou mais), onde o contexto longo começa a perder |

## 7. Como rodar

Requisitos: Python 3.12 ou mais novo e [`uv`](https://docs.astral.sh/uv/). A chave da API vem de `ANTHROPIC_API_KEY`, por variável de ambiente ou por um arquivo `.env` (copie o `.env.example`; o `.env` não vai para o git).

```
uv sync                                   # ambiente
uv run pytest                             # gate de testes (sem API)
uv run python -m memoria.passiva          # chat de terminal com o exemplo passivo
uv run python -m memoria.ativa            # chat de terminal com o exemplo ativo
uv run python -m memoria.avaliacao.rodar  # verificação completa (usa API, custa dinheiro)
```

- **Testes.** `uv run pytest` roda tudo com um cliente fake e nunca chama a API.
- **Chats de terminal.** `/nova` simula a troca de sessão (zera a janela e mantém a memória); Ctrl+D encerra (Ctrl+Z e Enter no Windows). A memória fica no arquivo `memoria.db`, fora do git; apague o arquivo para começar do zero. O chat passivo imprime S depois de cada turno; o ativo imprime cada chamada de ferramenta.
- **Verificação.** **Usa a API de verdade e custa dinheiro**: a rodada deste documento custou US$ 6,14, com o juiz, e levou cerca de 37 minutos. Para rodar uma configuração só, passe o nome dela: `uv run python -m memoria.avaliacao.rodar passiva` (ou `sem_memoria`, `contexto_completo`, `ativa`). Cada rodada grava uma pasta nova em `resultados/<AAAAMMDD-HHMMSS>/`.
- **Windows.** Com a saída redirecionada para arquivo ou pipe, o Python usa a codificação cp1252 e os acentos saem trocados. Defina `PYTHONUTF8=1` antes (no PowerShell, `$env:PYTHONUTF8 = "1"`).

## Referências

### Fundamento

- Du, P. (2026). Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv:2603.07670.
- Sumers, T. R. et al. (2024). Cognitive Architectures for Language Agents. TMLR.

### Memória de agentes

- Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.
- Zhang, Q. et al. (2026). Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models. ICLR.
- Rasmussen, P. et al. (2025). Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956.

### Avaliação

- Maharana, A. et al. (2024). Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). ACL.
- Wu, D. et al. (2025). LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR.
- Hu, Y., Wang, Y. e McAuley, J. (2026). Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions (MemoryAgentBench). ICLR.
- Flynt, J. (2026). Structured Belief State and the First Precision-Aware Evaluation of LLM Memory Retrieval (PrecisionMemBench). arXiv:2605.11325.
