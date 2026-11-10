"""Prompts da OAB em PT-BR e despacho de templates (OAB ou autor)."""
from replica.autor.templates import get_template, normalize_agent_name

SYSTEM_OAB = "Você é um assistente prestativo que lê o contexto e o memoriza para consultas futuras."
SYSTEM_SEM_MEMORIA = "Você é um assistente prestativo."
INSTRUCAO_LETRA = "Responda apenas com a letra da alternativa correta (A, B, C ou D) e não escreva mais nada."

# Imitam `ruler_qa` (H1) e `in_context_learning` (H2) do autor, inclusive o `\\n` literal do `memorize`.
BASE_TEMPLATES_OAB = {
    "oab_h1": {
        "system": SYSTEM_OAB,
        "memorize": 'Diálogo entre Usuário e Assistente {time_stamp}\\n<Usuário> O contexto a seguir são os documentos que li: \n{context}\n <Assistente> Aprendi os documentos e vou responder à questão que você fizer.',
        "query": {
            "long_context_agent": "Responda à questão com base nos documentos memorizados. " + INSTRUCAO_LETRA + " \n\nQuestão: {question} \n\n Resposta:",
            "rag_agent": "Responda à questão com base nos documentos memorizados. " + INSTRUCAO_LETRA + " \n\n Agora responda à questão: {question}",
        },
    },
    "oab_h2": {
        "system": SYSTEM_OAB,
        "memorize": 'Diálogo entre Usuário e Assistente {time_stamp} \\n<Usuário> O contexto a seguir são os exemplos que estudei: \n{context}\n <Assistente> Estudei os exemplos e vou responder à questão que você fizer.',
        "query": {
            "long_context_agent": "Use os exemplos do contexto, questões da prova da OAB com o gabarito, para responder à nova questão. Responda apenas \"Gabarito: {{letra}}\" e nada mais. \n\n{question} \n\n Gabarito:",
            "rag_agent": "Use os exemplos do contexto, questões da prova da OAB com o gabarito, para responder à nova questão. Responda apenas \"Gabarito: {{letra}}\" e nada mais. \n\nQuestão:{question} \n\n Gabarito:",
        },
    },
}
TEMPLATE_SEM_MEMORIA = "Responda à questão da prova da OAB. " + INSTRUCAO_LETRA + " \n\nQuestão: {question} \n\n Resposta:"


def obter_template(sub_dataset: str, template_name: str, agent_name: str) -> str:
    """Template da OAB para `oab_h1*` e `oab_h2*`; qualquer outro cai no código do autor."""
    for base in ("oab_h1", "oab_h2"):
        if sub_dataset.startswith(base):
            modelo = BASE_TEMPLATES_OAB[base][template_name]
            if isinstance(modelo, dict):
                return modelo[normalize_agent_name(agent_name)]
            return modelo
    return get_template(sub_dataset, template_name, agent_name)
