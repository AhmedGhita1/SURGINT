# SURGINT ontology 1.0.0

SURGINT records handling and inventory knowledge for the 13 labels emitted by the surgical-tray detector.

- `SURGINT.owx` is the OWL/XML serialization.
- `SURGINT.owl` is the RDF/XML serialization intended for direct use with `owlready2`.
- Ontology IRI: `http://www.semanticweb.org/ahmedghita/ontologies/2026/SURGINT`
- Version IRI: `http://www.semanticweb.org/ahmedghita/ontologies/2026/SURGINT/1.0.0`

The files express the same OWL 2 ontology. There are exactly 13 user-defined classes. Each is a direct subclass of `owl:Thing`, no detector class subclasses another detector class, and all 13 are declared mutually disjoint. Their untagged `rdfs:label` values are the exact detector join keys.

## Modeling pattern

Knowledge about a detector class is represented with an OWL `ObjectHasValue` restriction. For example, `scalpel SubClassOf hasHazardClass value sharp` means that every detected scalpel has the named individual `sharp` as its hazard value. The values are named and annotated individuals rather than string literals, so more annotations or values can be added later without changing the property model.

`hasDisposalStatus` and `hasHazardClass` are functional because an item in this deployment profile has at most one value for each controlled dimension. The controlled alternatives are explicitly different individuals; OWL does not otherwise make a unique-name assumption.

SURGINT follows the open-world assumption. An unasserted value means **unknown at this detector-label resolution**. It does not mean `none`, `false`, or the opposite value.

The individual `none` is narrowly defined as “no intrinsic sharp hazard asserted.” It does not mean that an item is sterile, uncontaminated, or free from biological, chemical, electrical, or other risk.

## Decisions by detector class

An em dash means that the property is deliberately unasserted.

| Detector label | `hasDisposalStatus` | `hasHazardClass` | `hasFunctionalRole` | Decision and justification |
|---|---|---|---|---|
| `scalpel` | — | `sharp` | `cutting` | A scalpel has an intrinsic cutting edge. Disposal is unasserted because the detector label does not separate a reusable handle, a replaceable single-use blade, and an all-in-one disposable scalpel. |
| `forceps` | — | — | `grasping` | Forceps can be reusable or single-use and can have blunt, toothed, or sharply pointed tips. The flat label changes neither ambiguity, so both handling properties remain unasserted. |
| `hemostat` | — | `none` | `clamping` | Reusable and single-use hemostats both exist, so disposal is unasserted. A conventional hemostat is a serrated clamping instrument without a cutting or puncturing edge, so its sharp-hazard value is `none`. |
| `scissors` | — | `sharp` | `cutting` | Surgical scissors have cutting edges and are treated as sharp. Reusable and single-use models share the same visual class, so disposal is unasserted. |
| `retractor` | — | — | `tissue-retraction` | Retractors can be reusable or single-use, and their working ends can be blunt or sharp-pronged. The detector class does not resolve either distinction. |
| `dissector` | — | — | `tissue-dissection` | Dissectors include reusable and single-use devices and both blunt and sharp working ends. The detector class does not resolve either distinction. |
| `syringe` | `single-use` | — | `fluid-delivery` | Single-use is asserted as required by the SURGINT deployment profile. Hazard is unasserted because the label does not distinguish a needleless syringe from a needle-bearing assembly. |
| `tray` | `reusable` | `none` | `instrument-support` | Reusable and non-sharp are explicit deployment-profile requirements. The role captures its use for holding and organizing instruments or supplies. |
| `basin` | `reusable` | `none` | `fluid-containment` | Reusable and non-sharp are explicit deployment-profile requirements. A basin receives or holds fluid. |
| `bowl` | `reusable` | `none` | `fluid-containment` | Reusable and non-sharp are explicit deployment-profile requirements. A bowl receives or holds fluid or supplies. |
| `gauze` | `single-use` | `none` | `absorption` | Single-use is an explicit deployment-profile requirement. Gauze is a soft good with no intrinsic sharp edge and is handled for absorption. |
| `bandage` | `single-use` | `none` | `wound-dressing` | Single-use is an explicit deployment-profile requirement. A bandage is a soft good with no intrinsic sharp edge and is handled as wound-dressing material. |
| `tube` | — | `none` | `fluid-conveyance` | Disposal is unasserted because both disposable tubing and validated reusable tubing exist and the detector cannot distinguish them. The flexible tube itself has no intrinsic sharp edge; a separate attached needle, trocar, or introducer must not be inferred from the `tube` label. |

The requirements that mark syringes, gauze, and bandages single-use and trays, basins, and bowls reusable define this deployment profile. They should not be read as universal claims about every product ever sold under those generic names.

## Additional property

`hasFunctionalRole` is the one property added beyond the required disposal and hazard properties. It is justified because role is stable at the granularity of each detector label and supports handling tasks such as tray organization, inventory grouping, and detecting a missing functional category without introducing a class hierarchy. Its values are named individuals and the property is intentionally not functional, allowing future instruments to carry multiple roles.

No material, sterility, contamination, reprocessing method, waste-stream, or count-policy property is asserted. Those facts vary by product, packaging state, use state, facility policy, or procedure and cannot be determined from these 13 labels alone.

## Reading class facts with owlready2

The facts are class restrictions, not object-property assertions about the class objects. A small reader can extract the `hasValue` restrictions as follows:

```python
from pathlib import Path
from owlready2 import Restriction, VALUE, World

path = Path("ontology/surgint/SURGINT.owl").resolve()
world = World()
onto = world.get_ontology(str(path)).load()


def asserted_values(detector_class, prop):
    return [
        restriction.value
        for restriction in detector_class.is_a
        if isinstance(restriction, Restriction)
        and restriction.type == VALUE
        and restriction.property == prop
    ]


classes_by_label = {
    str(cls.label.first()): cls
    for cls in onto.classes()
}

scalpel = classes_by_label["scalpel"]
print([value.iri for value in asserted_values(scalpel, onto.hasHazardClass)])
# ['http://www.semanticweb.org/ahmedghita/ontologies/2026/SURGINT#sharp']
```

Use the RDF/XML file for this reader. An OWL/XML copy is supplied for OWL tools and interchange.
