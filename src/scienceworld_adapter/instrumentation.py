"""Read-only latent-state coverage diagnostics, never a completeness certificate."""

import json


def declared_fields(value) -> list[dict]:
    fields = []
    cls = value.getClass()
    while cls is not None and str(cls.getName()) != "java.lang.Object":
        fields.extend({"declaring_class": str(cls.getName()), "name": str(f.getName()),
                       "type": str(f.getType().getName())} for f in cls.getDeclaredFields())
        cls = cls.getSuperclass()
    return sorted(fields, key=lambda f: (f["declaring_class"], f["name"]))


def coverage_snapshot(environment) -> dict:
    interface = environment.env.server.agentInterface().get()
    universe = interface.universe()
    iterator = universe.getContainedObjectsAndPortalsRecursive(True, True).iterator()
    objects = {str(universe.uuid()): universe}
    while iterator.hasNext():
        obj = iterator.next()
        objects[str(obj.uuid())] = obj
    records = []
    for key, obj in sorted(objects.items(), key=lambda kv: int(kv[0])):
        records.append({"uuid": key, "class": str(obj.getClass().getName()),
                        "json_export": json.loads(obj.toJSON()), "declared_fields": declared_fields(obj)})
    roots = {name: declared_fields(getattr(interface, name)()) for name in
             ("task", "inputParser", "objMonitor", "actionHandler")}
    roots["agentInterface"] = declared_fields(interface)
    return {"scope": "read_only_field_coverage_diagnostic_not_label_evidence", "complete": False,
            "reasons": ["field names/types are not full field values", "toJSON omits latent properties",
                        "goal/parser/RNG/static identifier state not serialized completely"],
            "objects_including_portals": records, "runtime_root_fields": roots,
            "engine_iteration": interface.getCurIterations()}
