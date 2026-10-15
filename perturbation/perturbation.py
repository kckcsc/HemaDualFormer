
import argparse
import pickle
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Classification checkpoint directory containing configuration and weights")
    parser.add_argument("--dataset", required=True, help="Tokenized dataset directory")
    parser.add_argument("--class-dictionary", required=True, help="Trusted pickle containing the matching class-ID dictionary")
    parser.add_argument("--output-dir", required=True, help="New or empty output directory")
    parser.add_argument("--prefix", default="GSE116256_AML_Healthy_GMP", help="Output filename prefix; change it for other analyses")
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--cell-types", nargs="+", default=["GMP", "GMP-like"])
    parser.add_argument("--state-column", default="health_status")
    parser.add_argument("--start-state", default="aml")
    parser.add_argument("--goal-state", default="health")
    parser.add_argument("--reference-max-cells", type=positive_int, default=1000)
    parser.add_argument("--perturb-max-cells", type=positive_int, default=2000)
    parser.add_argument("--reference-batch-size", type=positive_int, default=32)
    parser.add_argument("--perturb-batch-size", type=positive_int, default=16)
    parser.add_argument("--reference-nproc", type=positive_int, default=16)
    parser.add_argument("--perturb-nproc", type=positive_int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    class_dictionary = Path(args.class_dictionary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for path in (model_dir, dataset):
        if not path.is_dir():
            parser.error(f"Input directory does not exist: {path}")
    if not class_dictionary.is_file():
        parser.error(f"Class dictionary does not exist: {class_dictionary}")
    if not args.prefix or any(c in args.prefix for c in '/\\:*?"<>|') or args.prefix in (".", ".."):
        parser.error("Prefix must be a filename prefix without path separators or reserved characters")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error("Choose a new or empty output directory to avoid mixing perturbation runs")
    if args.start_state == args.goal_state:
        parser.error("Start and goal states must differ")

    # Only load pickle files from trusted sources.
    with class_dictionary.open("rb") as handle:
        id2cell = pickle.load(handle)
    if not isinstance(id2cell, dict) or not id2cell:
        parser.error("Class dictionary must be a non-empty dictionary")
    num_classes = len(id2cell)

    # Defer training-library imports so --help works without these dependencies.
    from geneformer import InSilicoPerturber, InSilicoPerturberStats, EmbExtractor

    state_dir = output_dir / "state_embeddings"
    perturb_dir = output_dir / "isp_runs"
    stats_dir = output_dir / "isp_stats"
    for path in (state_dir, perturb_dir, stats_dir):
        path.mkdir(parents=True, exist_ok=True)

    cell_states_to_model = {
        "state_key": args.state_column,
        "start_state": args.start_state,
        "goal_state": args.goal_state,
        "alt_states": [],
    }
    filter_data = {args.cell_type_column: args.cell_types}

    # Preserve the original reference-state extraction settings.
    # No model_version or reference pooling override was active in the source.
    embex = EmbExtractor(
        model_type="CellClassifier",
        num_classes=num_classes,
        filter_data=filter_data,
        max_ncells=args.reference_max_cells,
        emb_layer=0,
        summary_stat="exact_mean",
        forward_batch_size=args.reference_batch_size,
        nproc=args.reference_nproc,
    )
    state_embeddings = embex.get_state_embs(
        cell_states_to_model, str(model_dir), str(dataset), str(state_dir), args.prefix
    )

    # Delete individual gene tokens using the original Geneformer settings.
    # The interaction between cls mode and mean_pool is version-dependent.
    isp = InSilicoPerturber(
        perturb_type="delete",
        perturb_rank_shift=None,
        genes_to_perturb="all",
        combos=0,
        anchor_gene=None,
        model_type="CellClassifier",
        num_classes=num_classes,
        emb_mode="cls",
        cell_emb_style="mean_pool",
        filter_data=filter_data,
        cell_states_to_model=cell_states_to_model,
        state_embs_dict=state_embeddings,
        max_ncells=args.perturb_max_cells,
        emb_layer=0,
        forward_batch_size=args.perturb_batch_size,
        nproc=args.perturb_nproc,
    )
    isp.perturb_data(str(model_dir), str(dataset), str(perturb_dir), args.prefix)

    # Summarize shifts toward the target state.
    ispstats = InSilicoPerturberStats(
        mode="goal_state_shift",
        genes_perturbed="all",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=cell_states_to_model,
    )
    # Preserve the original positional None argument; consult the pinned API.
    ispstats.get_stats(str(perturb_dir), None, str(stats_dir), args.prefix)
    print("Reference state outputs:", state_dir)
    print("Perturbation intermediate outputs:", perturb_dir)
    print("Final statistics:", stats_dir)


if __name__ == "__main__":
    main()
