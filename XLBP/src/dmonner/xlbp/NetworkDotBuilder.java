package dmonner.xlbp;

import java.util.Set;

import dmonner.xlbp.compound.Compound;
import dmonner.xlbp.compound.IndirectWeightBank;
import dmonner.xlbp.compound.WeightBank;
import dmonner.xlbp.compound.WeightedCompound;
import dmonner.xlbp.layer.DownstreamLayer;
import dmonner.xlbp.layer.Layer;
import dmonner.xlbp.layer.UpstreamLayer;
import dmonner.xlbp.util.ListSet;

/**
 * @author dmonner
 */
public class NetworkDotBuilder
{
	private final Set<String> nodes;
	private final Set<String> edges;
	private final Network root;

	public NetworkDotBuilder(final Network net)
	{
		root = net;
		nodes = new ListSet<String>();
		edges = new ListSet<String>();
		processNet(net);
	}

	private void processCompound(final Compound comp)
	{
		if(comp instanceof WeightedCompound)
		{
			final WeightedCompound wcomp = (WeightedCompound) comp;
			for(int i = 0; i < wcomp.nUpstreamWeights(); i++)
				processWeightBank(wcomp.getUpstreamWeights(i));
		}

		for(final Component component : comp.getComponents())
		{
			if(component instanceof Compound)
				processCompound((Compound) component);
			else if(component instanceof Layer)
				processNode((Layer) component);
			else if(component instanceof Network)
				processNet((Network) component);
			else
				throw new IllegalArgumentException("Unhandled");
		}
	}

	private void processEdge(final Layer from, final Layer to)
	{
		edges.add(from.getName() + " -> " + to.getName() + " [style=dotted]");
	}

	private void processNet(final Network net)
	{
		for(final Component component : net.getComponents())
		{
			if(component instanceof Compound)
				processCompound((Compound) component);
			else if(component instanceof Layer)
				processNode((Layer) component);
			else if(component instanceof Network)
				processNet((Network) component);
			else
				throw new IllegalArgumentException("Unhandled");
		}
	}

	private void processNode(final Layer node)
	{
		nodes.add(node.getName());

		if(node instanceof UpstreamLayer)
		{
			final UpstreamLayer unode = (UpstreamLayer) node;
			for(int i = 0; i < unode.nDownstream(); i++)
				processEdge(unode, unode.getDownstream(i).asDownstreamLayer());
		}

		if(node instanceof DownstreamLayer)
		{
			final DownstreamLayer dnode = (DownstreamLayer) node;
			for(int i = 0; i < dnode.nUpstream(); i++)
				processEdge(dnode.getUpstream(i).asUpstreamLayer(), dnode);
		}
	}

	private void processWeightBank(final WeightBank wb)
	{
		if(wb instanceof IndirectWeightBank)
			edges.add(wb.getWeightOutput() + " -> " + wb.getWeightInput() + " [style=dotted]");
		else
			edges.add(wb.getWeightOutput() + " -> " + wb.getWeightInput() + " [style=bold]");
	}

	@Override
	public String toString()
	{
		final StringBuilder sb = new StringBuilder();

		sb.append("digraph D {\n");
		sb.append("  rankdir = BT;\n");

		for(final String node : nodes)
		{
			sb.append("  ");
			sb.append(node);
			sb.append(";\n");
		}

		for(final String edge : edges)
		{
			sb.append("  ");
			sb.append(edge);
			sb.append(";\n");
		}

		sb.append(toSubgraphString(root, "  "));

		sb.append("}\n");

		return sb.toString();
	}

	private String toSubgraphString(final Compound net, final String pre)
	{
		final StringBuilder sb = new StringBuilder();
		final String name = net.getName();

		sb.append(pre);
		sb.append("subgraph cluster_");
		sb.append(name);
		sb.append(" {\n");
		sb.append(pre);
		sb.append("  ");
		sb.append("label=\"");
		sb.append(name);
		sb.append("\";\n");

		if(net instanceof WeightedCompound)
		{
			final WeightedCompound wcomp = (WeightedCompound) net;
			for(int i = 0; i < wcomp.nUpstreamWeights(); i++)
			{
				final WeightBank wb = wcomp.getUpstreamWeights(i);
				sb.append(toSubgraphString(wb.getWeightOutput(), pre));
				sb.append(toSubgraphString(wb.getWeightInput(), pre));
			}
		}

		for(final Component comp : net.getComponents())
		{
			if(comp instanceof Network)
				sb.append(toSubgraphString((Network) comp, pre + "  "));
			else if(comp instanceof Compound)
				sb.append(toSubgraphString((Compound) comp, pre + "  "));
			else if(comp instanceof Layer)
				sb.append(toSubgraphString((Layer) comp, pre + "  "));
			else
				throw new IllegalArgumentException("Unhandled");
		}

		sb.append(pre);
		sb.append("}\n");

		return sb.toString();
	}

	private String toSubgraphString(final Layer layer, final String pre)
	{
		final StringBuilder sb = new StringBuilder();
		final String name = layer.getName();

		sb.append(pre);
		sb.append(name);
		sb.append(";\n");

		return sb.toString();
	}

	private String toSubgraphString(final Network net, final String pre)
	{
		final StringBuilder sb = new StringBuilder();
		final String name = net.getName();

		sb.append(pre);
		sb.append("subgraph cluster_");
		sb.append(name);
		sb.append(" {\n");
		sb.append(pre);
		sb.append("  ");
		sb.append("label=\"");
		sb.append(name);
		sb.append("\";\n");

		for(final Component comp : net.getComponents())
		{
			if(comp instanceof Network)
				sb.append(toSubgraphString((Network) comp, pre + "  "));
			else if(comp instanceof Compound)
				sb.append(toSubgraphString((Compound) comp, pre + "  "));
			else if(comp instanceof Layer)
				sb.append(toSubgraphString((Layer) comp, pre + "  "));
			else
				throw new IllegalArgumentException("Unhandled");
		}

		sb.append(pre);
		sb.append("}\n");

		return sb.toString();
	}
}
