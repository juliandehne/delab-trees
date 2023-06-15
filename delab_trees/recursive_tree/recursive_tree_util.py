def solve_orphans(orphans, tree_node):
    if len(orphans) == 0:
        return False, []

    rest_orphans = []
    orphan_added = False
    for orphan in orphans:
        added = tree_node.find_parent_of(orphan)
        # either or the list es reduced by one element
        if not added:
            rest_orphans.append(orphan)
        # either nothing was found that False is returned stopping the cycle
        else:
            orphan_added = True
    if len(orphans) == len(rest_orphans):
        # print(f"could not reconstruct faulty tree for conversation_id {orphans[0].data['conversation_id']}")
        # print("could not reconstruct faulty tree with {} orphans ".format(rest_orphans))
        return False, rest_orphans
    assert len(orphans) != len(rest_orphans)
    return orphan_added, rest_orphans
