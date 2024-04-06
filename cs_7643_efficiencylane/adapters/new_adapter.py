from transformers import RobertaModelWithHeads, AdapterConfig

model = RobertaModelWithHeads.from_pretrained("roberta-base")
adapter_name = model.load_adapter("adapter-hub/roberta-base-pf/cs_domain", source="hf", config=AdapterConfig(mh_adapter=True))


citation_intent_config = AdapterConfig(mh_adapter=True)
relation_classification_config = AdapterConfig(mh_adapter=True)

citation_intent_adapter = model.add_adapter("citation_intent_adapter", config=citation_intent_config)

model.train_adapter(["citation_intent_adapter"])
