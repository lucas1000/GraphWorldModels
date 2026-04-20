// Neo4j schema setup — constraints and indexes for the graph world model.
// Run once after starting Neo4j:
//   cat schema.cypher | cypher-shell -u neo4j -p password

// Uniqueness constraints
CREATE CONSTRAINT state_id IF NOT EXISTS FOR (s:State) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Indexes for fast temporal slicing
CREATE INDEX state_tick IF NOT EXISTS FOR (s:State) ON (s.tick);
CREATE INDEX state_room IF NOT EXISTS FOR (s:State) ON (s.room_id);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
