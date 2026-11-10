# Conferência da reclassificação de matéria: rodada 2 (aprovada)

Conferência **provisória** feita pelo Claude (controlador da execução do plano) em 2026-11-10, porque a sessão rodou sem o Pedro; o Pedro reconfere. Amostra nova (semente 43), depois do ajuste do prompt que passou a informar o número da questão e a ordem dos blocos da prova da FGV. Resultado: **0 erros em 30**, dentro do limite de 3. A rodada 1 está em `docs/conferencia-materias-rodada1.md`.

Ressalva: 65 das 953 questões (6,8%) ficaram com matéria fora do bloco da FGV, e nenhuma delas caiu nesta amostra. Várias repetem erros da rodada 1 (por exemplo 2018-03-41, 2022-01-41 e 2019-01-31). Como o contexto longo e o RAG recebem a mesma lei em cada questão da H1, esse ruído afeta os dois braços igualmente e não enviesa a comparação pareada; o documento final registra a limitação.

| id | trecho (150 caracteres) | rótulo do banco | rótulo novo | ok? |
|---|---|---|---|---|
| 2018-02-49 | Antes da decretação de falência da sociedade Talismã & Sandolândia Ltda., foi ajudada aqui a execução por título extrajudicial por Frigorífico Rio Son | Direito Processual Civil | Direito Empresarial | sim |
| 2019-01-73 | Rita de Cássia é enfermeira em um hospital desde 10/01/2018, no qual trabalhava em regime de escala de 12x36 horas, no horário das 7.00 às 19.00 horas | Direito do Trabalho | Direito do Trabalho | sim |
| 2022-01-6 | Beatriz, advogada, oferece representação perante a OAB em razão de Isabela, outra advogada que atua na mesma área e na mesma cidade, ter supostamente  | Ética Profissional | Ética Profissional | sim |
| 2022-02-53 | João ajuizou ação de indenização por danos materiais e morais contra Carla. Ao examinar a petição inicial, o juiz competente entendeu que as causas di | Direito Processual Civil | Direito Processual Civil | sim |
| 2018-25_1 | Lina, cidadã que não exerce a advocacia, deseja endereçar à presidência de certa Subseção da OAB representação pela instauração de processo disciplina | Ética Profissional | Ética Profissional | sim |
| 2020-01-40 | Arnaldo faleceu e deixou os filhos Roberto e Álvaro. No inventário judicial de Arnaldo, Roberto, devedor contumaz na praça, renunciou à herança, em 05 | Ética Profissional | Direito Civil | sim |
| 2019-03-13 | As chuvas torrenciais que assolaram as regiões Norte e Nordeste do país resultaram na paralisação de serviços públicos essenciais ligados às áreas de  | Direito Constitucional | Direito Constitucional | sim |
| 2022-01-37 | Bento Albuquerque contou ao intuito de realizar o sonho de passar a aposentadoria na beira da praia, resolveu iniciar Monique, proprietária do imóvel  | Direito Civil | Direito Civil | sim |
| 2022-01-62 | Durante uma festa de confraternização entre amigos da faculdade, em 19 de junho de 2020, começou uma discussão entre Plínio e Carlos, tendo a mãe de P | Direito Penal | Direito Penal | sim |
| 2022-03-16 | Um órgão público, detentor de banco de dados com informações passíveis de serem transmitidas a terceiros, possui informações inexatas a respeito de Jo | Direito Constitucional | Direito Constitucional | sim |
| 2018-03-36 | Arnaldo foi procurado por sua irmã Zulmira, que lhe ofereceu R$ 1 milhão para adquirir o apartamento que ele possui na orla da praia. Receoso, no enta | Direito Civil | Direito Civil | sim |
| 2020-01-32 | Rafael, funcionário da concessionária prestadora do serviço público de fornecimento de água canalizada, realizava reparo na rede subterrânea, quando d | Ética Profissional | Direito Administrativo | sim |
| 2021-02-4 | Anderson, advogado, decidiu renunciar ao mandato outorgado por Adriana. Nessa hipótese, segundo o Estatuto da Advocacia e da OAB, é correto afirmar qu | Ética Profissional | Ética Profissional | sim |
| 2021-01-14 | Ante a ausência de norma regulamentadora de direito social na Constituição da República, cuja edição é de competência da União, ao que se soma a const | Direito Constitucional | Direito Constitucional | sim |
| 2021-02-49 | Farmácias Mundo Novo Ltda. é locatária de um imóvel não residencial onde funcionam uma de suas filiais. No curso da vigência do contrato, que se encon | Direito Civil | Direito Empresarial | sim |
| 2018-02-3 | O advogado Fred dirigiu-se, em certa ocasião, a uma delegacia de polícia e a um presídio, a fim de entrevistar clientes seus que se encontravam, respe | Ética Profissional | Ética Profissional | sim |
| 2021-01-29 | O Município Alfa pretende formalizar uma parceria público-privada para a realização de obras, instalação de postes e prestação de serviços de iluminaç | Ética Profissional | Direito Administrativo | sim |
| 2020-01-12 | Josué, deputado federal no regular exercício do mandato, em entrevista dada em sua residência, à revista Pensamento, acusa sua adversária política Ali | Direito Constitucional | Direito Constitucional | sim |
| 2021-02-19 | Você, como advogado(a) atuante na defesa dos Direitos Humanos, foi convidado(a) para participar de um programa de debate na rádio local sobre a questã | Direito Constitucional | Direitos Humanos | sim |
| 2019-03-16 | Giuseppe, italiano, veio ainda criança para o Brasil, juntamente com seu país. Desde então, sofreu qualquer tipo de condenação penal, constituiu famíl | Direito Constitucional | Direito Constitucional | sim |
| 2021-02-63 | Vitor foi condenado pela prática de um crime de lesão corporal leve no contexto da violência doméstica e familiar contra a mulher, sendo aplicada pena | Direito Penal | Direito Penal | sim |
| 2022-03-74 | A sociedade empresária Mangiare Bene, do ramo de serviços de alimentação, tem um plano de expansão em que pretende assumir as atividades de outros res | Direito Civil | Direito do Trabalho | sim |
| 2021-01-6 | O advogado Filipe, em razão de sua notoriedade na atuação em defesa das minorias, foi procurado por representantes de certa pessoa jurídica X, que sol | Ética Profissional | Ética Profissional | sim |
| 2023-01-34 | A sociedade empresária Alfo é fabricante e comerciante de pilhas e baterias. Em matéria de responsabilidade compartilhada pelo ciclo de vida dos produ | Direito Civil | Direito Ambiental | sim |
| 2022-02-47 | Riqueza Comércio de Artigos Eletrônicos Ltda. sofreu duplicata na modalidade cartular em face de Papelaria Sul Brasil Ltda., que devidamente aceita, c | Direito Civil | Direito Empresarial | sim |
| 2020-01-14 | Alfa, entidade de classe de abrangência regional, legalmente constituída e em funcionamento há mais de 1 ano, ingressa, perante o Supremo Tribunal Fed | Direito Processual Civil | Direito Constitucional | sim |
| 2019-03-36 | Vilmar, produtor rural, possui contratos de compra e venda de milho com diversos pequenos proprietários. Com o intuito de adquirir novos insumos, Vilm | Direito Civil | Direito Civil | sim |
| 2018-25_4 | O advogado Valter instalou, na fachada do seu escritório, um discreto painel luminoso com os dizeres “Advocacia Trabalhista”. A sociedade de advogados | Ética Profissional | Ética Profissional | sim |
| 2018-02-68 | Um Delegado de Polícia, ao tomar conhecimento de um suposto crime de ação pública incondicionada, determina, de ofício, a instauração de inquérito pol | Direito Constitucional | Direito Processual Penal | sim |
| 2022-03-73 | Lúcio Lima foi contratado para trabalhar em uma empresa no ramo da construção civil. Seu empregador descobriu inúmeras erros trabalhistas, e, notadame | Direito Civil | Direito do Trabalho | sim |
