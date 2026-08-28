from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.models import ReleaseCreate, TestCandidate, TestQueryRequest, TestSessionCreate
from app.pipeline import RagService


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the supplied FAQ CSV and optionally publish it.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    service = RagService(get_settings())
    try:
        upload = await service.accept_upload(
            content=args.path.read_bytes(),
            filename=args.path.name,
            mime_type="text/csv",
            metadata={"knowledgeBaseId": "main-business-kb", "title": args.path.stem, "uploadedBy": "cli"},
            request_id="cli-upload-" + args.path.stem,
        )
        while True:
            await asyncio.sleep(1)
            current = service.get_document(upload.documentId)
            if current.status not in {"PROCESSING", "UPLOADED"}:
                break
        print(json.dumps(current.model_dump(mode="json"), ensure_ascii=False, indent=2))
        if current.status != "READY_FOR_TEST":
            raise SystemExit(2)
        session = service.create_test_session(
            TestSessionCreate(
                requestId="cli-session-" + upload.documentId,
                knowledgeBaseId="main-business-kb",
                mode="PRE_RELEASE",
                candidateDocuments=[TestCandidate(documentId=upload.documentId, documentVersion=current.documentVersion)],
                operatorId="cli",
            )
        )
        print(json.dumps(session.model_dump(mode="json"), ensure_ascii=False, indent=2))
        if args.publish:
            await service.test_query(
                session.testSessionId,
                TestQueryRequest(
                    requestId="cli-review-" + upload.documentId,
                    question="港澳居民如何注册内地微信账号？",
                ),
            )
            release = await service.start_release(
                ReleaseCreate(
                    requestId="cli-release-" + upload.documentId,
                    testSessionId=session.testSessionId,
                    candidateDocuments=[TestCandidate(documentId=upload.documentId, documentVersion=current.documentVersion)],
                    publishedBy="cli",
                    publishNote="Initial FAQ import",
                )
            )
            while True:
                await asyncio.sleep(1)
                current_release = service.release(release.releaseId)
                if current_release.status != "BUILDING":
                    break
            print(json.dumps(current_release.model_dump(mode="json"), ensure_ascii=False, indent=2))
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
