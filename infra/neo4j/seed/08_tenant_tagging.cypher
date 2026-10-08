// Tenant tagging — this seed graph is the Rampur Processing Unit, tenant
// "plant-mumbai-01". Stamp site_id on every tenant-scoped node so query-time
// isolation (retrieval.graph_traversal / agents.lessons_learned) actually
// bites. coalesce() makes this idempotent and non-destructive: a node that
// already carries an explicit site_id (e.g. loaded from a connector for
// another plant) is left untouched.
//
// Add a second plant's nodes with site_id:'plant-jamnagar-02' to exercise
// cross-tenant isolation end-to-end (see tests/retrieval/test_tenant_isolation.py
// for the enforced behavior).

MATCH (n)
WHERE n:Equipment OR n:FailureEvent OR n:WorkOrder
   OR n:RegulatoryClause OR n:Procedure OR n:Chunk OR n:Person OR n:Document
SET n.site_id = coalesce(n.site_id, 'plant-mumbai-01');
