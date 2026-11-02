"""Consume transaction IDs, score with the shared scorer, and emit fraud-scores."""
from __future__ import annotations

import argparse
import json
from typing import Any

from confluent_kafka import Consumer, KafkaError, Producer

from src.common.config import (
    KAFKA_BOOTSTRAP_DOCKER,
    KAFKA_BOOTSTRAP_HOST,
    KAFKA_FRAUD_SCORES_TOPIC,
    KAFKA_TRANSACTIONS_TOPIC,
)
from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer, scorer_from_environment


def process_transaction(
    payload: dict[str, Any],
    scorer: Any,
    repository: ScoreRepository,
    output_producer: Any,
    output_topic: str = KAFKA_FRAUD_SCORES_TOPIC,
) -> dict[str, Any]:
    transaction_id = int(payload["transaction_id"])
    result = repository.save_score(scorer.score(transaction_id))
    output_producer.produce(
        output_topic,
        key=str(transaction_id),
        value=json.dumps(result, separators=(",", ":")),
    )
    output_producer.poll(0)
    outstanding = output_producer.flush(30)
    if outstanding:
        raise RuntimeError(f"Output producer still has {outstanding} message(s) queued")
    return result


def consume_forever(
    consumer: Any,
    output_producer: Any,
    scorer: Any,
    repository: ScoreRepository,
    input_topic: str = KAFKA_TRANSACTIONS_TOPIC,
    output_topic: str = KAFKA_FRAUD_SCORES_TOPIC,
) -> None:
    consumer.subscribe([input_topic])
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            error = message.error()
            if error:
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(f"Kafka consumer error: {error}")
            payload = json.loads(message.value().decode("utf-8"))
            process_transaction(payload, scorer, repository, output_producer, output_topic)
            consumer.commit(message=message, asynchronous=False)
    finally:
        consumer.close()
        output_producer.flush(10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="use the docker-network Kafka address")
    parser.add_argument("--bootstrap-server", default=None)
    parser.add_argument("--group-id", default="fraud-scoring-v1")
    parser.add_argument("--input-topic", default=KAFKA_TRANSACTIONS_TOPIC)
    parser.add_argument("--output-topic", default=KAFKA_FRAUD_SCORES_TOPIC)
    parser.add_argument("--database")
    args = parser.parse_args()
    bootstrap = args.bootstrap_server or (KAFKA_BOOTSTRAP_DOCKER if args.docker else KAFKA_BOOTSTRAP_HOST)
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": args.group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    producer = Producer({"bootstrap.servers": bootstrap, "client.id": "fraud-score-publisher"})
    repository = ScoreRepository() if args.database is None else ScoreRepository(args.database)
    scorer = scorer_from_environment()
    print(json.dumps({"bootstrap_server": bootstrap, "input_topic": args.input_topic, "output_topic": args.output_topic}))
    consume_forever(consumer, producer, scorer, repository, args.input_topic, args.output_topic)


if __name__ == "__main__":
    main()
